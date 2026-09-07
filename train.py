import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Reduce CUDA memory fragmentation: cap the cache block size at 128MB; larger allocations are released immediately
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from utils.dataset import S3MDHRDataset
import math
from tensorboardX import SummaryWriter
import clip
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from torchvision import transforms
from utils.metrics import psnr, sam, rmse, ergas
from models.SMDHR import SMDHR_b

# RTX 5090 (Blackwell) + stable PyTorch setup
# Enable cuDNN benchmark to speed up convolution
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# Print a log message and also save it to a file
def plog(msg, savename):
    print(msg)
    with open(f"log/mylog-{savename}.txt", "a") as fp:
        fp.write(msg + "\n")


# Convert a tuple to a string
def tuple2str(t):
    return "-".join([str(i) for i in t])


# Randomly generate crop parameters
def randCropParams(x, size=112):
    i, j, h, w = transforms.RandomCrop.get_params(x, output_size=(size, size))
    # Random horizontal flip
    hf = False
    if random.random() > 0.5:
        hf = True
    # Random vertical flip
    vf = False
    if random.random() > 0.5:
        vf = True
    return i, j, h, w, hf, vf


# Crop the image using the given crop parameters
def cropImg(x, i, j, h, w, hf, vf, device='cuda'):
    x = TF.crop(x, i, j, h, w)
    if hf:
        x = TF.hflip(x)
    if vf:
        x = TF.vflip(x)
    return x.to(device)


def get_target_routing_distribution(degra_types, num_experts=10, device='cuda'):
    """
    Updated version: soft-label logic that asymmetrically guides the bm type
    (weak guidance to n, strong guidance to clean)
    """
    batch_size = len(degra_types)
    targets = torch.zeros((batch_size, num_experts), device=device)

    for i, d_type in enumerate(degra_types):
        d_str = str(d_type).lower()

        # Split exactly to avoid substring mismatches (e.g. h_b must not trigger h_b_n)
        parts = d_str.split('_')

        # Determine which broad degradation categories the current image has
        has_n = 'noise' in d_str or 'impulse' in d_str or 'n' in parts
        has_h = 'haze' in d_str or 'h' in parts
        has_b = 'blur' in d_str or 'gauss' in d_str or 'b' in parts
        has_bm = 'stripe' in d_str or 'bandwise' in d_str or 'partial' in d_str or 'complete' in d_str or 'bm' in parts

        # Initialize the base expert weight scores of the current sample
        expert_weights = np.zeros(num_experts)

        # 1. Standard degradations: assign the standard activation weight of 1.0
        if has_n:
            expert_weights[0:3] += 1.0  # E0, E1, E2 (denoising)
        if has_h:
            expert_weights[3:6] += 1.0  # E3, E4, E5 (dehazing)
        if has_b:
            expert_weights[6:9] += 1.0  # E6, E7, E8 (deblurring)

        # 2. Special handling for bm
        if has_bm:
            if not has_n:
                # If only bm is present (no pure noise): guide weakly to n, strongly to clean
                # The 0.3 and 1.5 here are relative weights; tune them to your actual load
                expert_weights[0:3] += 0.3  # weak pull: +0.1 to each of E0, E1, E2 (0.3 in total)
                expert_weights[9] += 0.3  # strong pull: +1.5 to E9 (Clean)
            else:
                # If both n and bm are present, the n experts were already activated above.
                # To keep n from staying overloaded, divert a little extra weight to clean.
                expert_weights[9] += 0.0

                # 3. Fallback for purely clean images
        if not (has_n or has_h or has_b or has_bm):
            expert_weights[9] = 1.0

        # 4. Normalize the weights into a probability distribution
        total_weight = np.sum(expert_weights)
        if total_weight > 0:
            expert_weights = expert_weights / total_weight

        targets[i] = torch.tensor(expert_weights, dtype=torch.float32, device=device)

    return targets


def maml_inner_update(model, support_data, inner_lr, inner_steps):
    fast_weights = []
    param_names = []
    for name, param in model.k_predictor.named_parameters():
        if param.requires_grad:
            fast_weights.append(param.clone().requires_grad_(True))
            param_names.append(name)

    original_params = {name: param.data.clone() for name, param in model.k_predictor.named_parameters()}

    for step in range(inner_steps):
        with torch.enable_grad():
            for (name, param), new_param in zip(model.k_predictor.named_parameters(), fast_weights):
                param.data = new_param.data

            support_loss = compute_support_loss(model, support_data)
            print(f"Step {step + 1}, Support Loss: {support_loss.item()}")

            for name, param in model.k_predictor.named_parameters():
                param.data = original_params[name]

        grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True, allow_unused=True)

        for name, grad in zip(param_names, grads):
            if grad is None:
                print(f"Warning: Gradient for parameter {name} is None")
            else:
                print(f"Gradient for {name}: {grad.norm().item()}")

        updated_fast_weights = []
        for w, g, name in zip(fast_weights, grads, param_names):
            if g is None:
                print(f"Skipping update for {name} due to None gradient")
                updated_fast_weights.append(w)
            else:
                updated_fast_weights.append(w - inner_lr * g)
        fast_weights = updated_fast_weights

    return fast_weights, original_params


def compute_support_loss(model, support_data):
    """Compute the support-set loss - simplified version"""
    model.eval()  # Note: stay in eval mode here while still allowing gradient computation

    x, gt = support_data["source"], support_data["target"]
    x = x.to('cuda').float()
    gt = gt.to('cuda').float()

    # Forward pass
    output = model(x)

    # Compute the loss - adapt it to your actual loss function
    loss = F.l1_loss(output, gt)  # example loss function
    return loss

# Training function
def trainer(args, only_val=False):
    ## Read the data files
    print(f'Interpolation: {args.intp}, Long Prompt: {args.long}')
    device = 'cuda'

    # Create the training data loader
    train_loader = torch.utils.data.DataLoader(
        S3MDHRDataset(args.root, img_size=args.cropsize, long_prompt=args.long, mode="train",
                             interpolate=args.intp),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    # Create the validation data loader
    val_loader = torch.utils.data.DataLoader(
        S3MDHRDataset(args.root, img_size=args.imgsize, long_prompt=args.long, mode="test",
                             interpolate=args.intp),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load the CLIP model and set it to evaluation mode
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model_clip.eval()

    savename = args.savename
    img_size = (args.imgsize, args.imgsize)
    win_size = tuple(args.win_size)
    patch_size = tuple(args.pat_size)
    n_layers = tuple(args.n_layers)

    # Create the model
    model = eval(savename)()
    model.cuda()
    print(args.savename, args.state_dict)

    state_dict = None
    lr = args.lr

    # Create the optimizer - keep routing module parameters separate
    all_params = list(model.parameters())
    # Changed: also add the degradation_router parameters to router_params
    router_params = list(model.k_predictor.parameters()) + list(model.degradation_router.parameters())
    main_params = [p for p in all_params if p not in set(router_params)]

    # Create the optimizer groups
    optimizer = optim.AdamW([
        {'params': main_params, 'lr': lr},
        {'params': router_params, 'lr': lr}
    ])

    # Load the pretrained model state
    if args.state_dict is not None:
        state_dict = torch.load(args.state_dict, weights_only=False)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])

    # Training phase definitions
    ROUTER_PRETRAIN_EPOCHS = 0  # router pretraining phase
    META_LEARNING_EPOCHS = 0  # meta-learning phase
    JOINT_TRAINING_EPOCHS = args.epochs - ROUTER_PRETRAIN_EPOCHS - META_LEARNING_EPOCHS

    # Meta-learning parameters
    INNER_LR = 0.001  # inner-loop learning rate
    INNER_STEPS = 3  # number of inner-loop steps
    META_BATCH_SIZE = 4  # meta-batch size

    model.train()

    # Define the loss function and the TensorBoard writer
    writer = SummaryWriter(f"log/tensorboard-{savename}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5, last_epoch=-1)

    # Create the directories that hold checkpoints and logs
    if not os.path.isdir("ckpt"):
        os.mkdir("ckpt")
    if not os.path.isdir("log"):
        os.mkdir("log")

    resume_ind = 0 if state_dict is None else state_dict["epoch"]
    step = resume_ind
    best_sam = math.inf if state_dict is None else state_dict["sam"]

    # ============ Routing supervision and evaluation setup ============
    routing_info_list = []

    def shared_routing_hook(module, input, output):
        # Intercept the routing network outputs: gates, top_k_indices, top_k_values
        gates, top_k_indices, top_k_values = output
        # Keep both gates (used for the loss) and top_k_indices (used for utilization stats)
        routing_info_list.append({
            'gates': gates,
            'top_k_indices': top_k_indices.detach().cpu()
        })

    # Attach hooks to every RoutingFunction module
    routing_hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RoutingFunction':
            routing_hooks.append(module.register_forward_hook(shared_routing_hook))
    # =======================================================

    # Training loop
    for epoch in range(resume_ind + 1, args.epochs + 1):
        # Clear GPU memory fragmentation at the start of each epoch
        torch.cuda.empty_cache()

        running_loss, running_loss1, running_loss2, running_loss3, running_loss4 = 0.0, 0.0, 0.0, 0.0, 0.0
        running_route_loss = 0.0

        # ============== Determine the training phase ==============
        if epoch <= 100:
            # Phase 3: joint training for the first 100 epochs (all parameters stay trainable)
            phase = "joint_training"
            for param in all_params:
                param.requires_grad = True
        else:
            # Phase 4: joint training after epoch 100 (freeze k_predictor and degradation_router)
            phase = "joint_training_frozen_router"
            for param in main_params:
                param.requires_grad = True
            for param in router_params:
                param.requires_grad = False

        # Dynamically adjust the spectral continuity loss weight
        lambda_cont = max(0.001, min(0.5, 0.001 * (epoch // 10)))  # starts at 0.1, +0.1 every 30 epochs

        # Initialize the tqdm progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{args.epochs} [{phase}]",
            unit="batch",
        )

        if not only_val:
            for batch_idx, (data) in pbar:
                try:
                    # ============== Meta-learning task sampling ==============
                    if phase == "meta_learning" and batch_idx % META_BATCH_SIZE == 0:
                        meta_support_data = []
                        for _ in range(META_BATCH_SIZE):
                            try:
                                meta_support_data.append(next(train_iter))
                            except:
                                train_iter = iter(train_loader)
                                meta_support_data.append(next(train_iter))
                    # ============== Meta-learning inner loop ==============
                    # Meta-learning branch inside the training loop
                    if phase == "meta_learning":
                        fast_weights, original_params = maml_inner_update(model,
                                                                          meta_support_data[batch_idx % META_BATCH_SIZE],
                                                                          INNER_LR, INNER_STEPS)

                        current_params = {name: param.data.clone() for name, param in
                                          model.k_predictor.named_parameters()}

                        for (name, param), new_param in zip(model.k_predictor.named_parameters(), fast_weights):
                            param.data = new_param.data


                    # Prepare the training data
                    x, t, gt = data["source"], data["degra_type"], data["target"]
                    x, t, gt, ori = data["source"], data["degra_type"], data["target"], data["ori"]
                    optimizer.zero_grad()

                    x = x.to(device)
                    gt = gt.to(device)
                    ori = ori.to(device)

                    # Clear stale hook data left by meta-learning or other forward passes
                    routing_info_list.clear()

                    # Forward pass
                    output, loss1, loss2, loss3, loss4 = model(x, ori, gt)

                    # ========== Compute the routing guidance loss (Routing Loss) ==========
                    # Target distribution over E0-E8 only (num_experts=9)
                    target_gates = get_target_routing_distribution(t, num_experts=10, device=device)
                    route_loss = 0.0

                    if len(routing_info_list) > 0:
                        for info in routing_info_list:
                            # MSE-supervise only the first 9 gate probabilities, ignoring the last Clean expert
                            route_loss += F.mse_loss(info['gates'], target_gates)
                        route_loss = route_loss / len(routing_info_list)
                    # ==================================================================

                    # Compute the individual reconstruction losses
                    loss1 = loss1.sum()
                    loss2 = loss2.sum()
                    loss3 = loss3.sum()
                    loss4 = loss4.sum()

                    # Set the routing loss weight (suggested range 1.0 ~ 5.0)
                    lambda_route = 0.001

                    loss = loss1 + 0.1 * loss2 + 0.01 * loss3 + 0.01 * loss4 + lambda_route * route_loss

                    # Backpropagation
                    loss.backward()

                    # Gradient clipping: prevent numerical instability on the RTX 5090 nightly build
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # CUDA sync check: catch CUDA errors early on the nightly build
                    if batch_idx % 10 == 0:
                        torch.cuda.synchronize()

                    # Meta-learning outer-loop update
                    if phase == "meta_learning":
                        # Restore the original parameters
                        for name, param in model.k_predictor.named_parameters():

                            param.data = original_params[name]
                        # Outer-loop update
                        optimizer.step()
                        optimizer.zero_grad()

                    # Standard optimizer step for non-meta-learning phases (one update per batch)
                    if phase != "meta_learning":
                        optimizer.step()
                    running_loss += loss.item()
                    running_loss1 += loss1.item() if not phase == "router_pretrain" else 0
                    running_loss2 += loss2.item() if not phase == "router_pretrain" else 0
                    running_loss3 += loss3.item() if not phase == "router_pretrain" else 0
                    running_loss4 += loss4.item() if not phase == "router_pretrain" else 0

                    # Added: accumulate the routing loss safely
                    running_route_loss += route_loss.item() if isinstance(route_loss, torch.Tensor) else route_loss

                    # Update the progress bar description
                    pbar.set_description(f"Epoch {epoch}/{args.epochs} [{phase}] Loss: {loss.item():.4f}")

                except Exception as e:
                    plog(f"[FATAL] Training crashed at epoch {epoch}, batch {batch_idx}: {type(e).__name__}: {e}", savename)
                    import traceback
                    plog(traceback.format_exc(), savename)
                    raise


        # Validation
        if epoch % 5 == 0 and epoch >= 1:
            # Start validation
            print("Start Validation")
            val_expert_tracker = {}
            with torch.no_grad():
                rmses, sams, l1loss, fnames, psnrs, ergass = [], [], [], [], [], []
                start_time = time.time()
                for ind2, (data) in enumerate(val_loader):
                    try:
                        vx, vt, vgt, vfn, vori = data["source"], data["degra_type"], data["target"], data["filename"], data["ori"]
                        model.eval()
                        vx = vx.to(device)
                        vgt = vgt.to(device)
                        vori = vori.to(device)

                        # Clear hook data from the previous forward pass to prevent GPU tensor accumulation
                        routing_info_list.clear()

                        # Forward pass (skip vgt to avoid extra losses and reduce memory usage)
                        val_dec = model(vx, vori)

                        # ============== Track expert utilization ==============
                        d_type_raw = vt[0]
                        d_str = str(d_type_raw).lower()
                        mapped_parts = []

                        # Extract keywords and map them to h, b, n, bm
                        if 'haze' in d_str:
                            mapped_parts.append('h')
                        if 'blur' in d_str or 'gauss' in d_str:
                            mapped_parts.append('b')
                        if 'noise' in d_str or 'impulse' in d_str:
                            mapped_parts.append('n')
                        # Group all mask/stripe/missing-type keywords under bm
                        if 'stripe' in d_str or 'bandwise' in d_str or 'partial' in d_str or 'complete' in d_str or 'bm' in d_str:
                            mapped_parts.append('bm')

                        # No degradation matched: name it clean; otherwise join the parts with underscores
                        if not mapped_parts:
                            simplified_d_type = 'clean'
                        else:
                            simplified_d_type = "_".join(mapped_parts)

                        # Use the simplified category label as the dictionary key
                        if simplified_d_type not in val_expert_tracker:
                            val_expert_tracker[simplified_d_type] = {'counts': np.zeros(10), 'calls': 0}

                        for info in routing_info_list:
                            top_k = info['top_k_indices']  # shape: (1, K)
                            for b in range(top_k.size(0)):
                                for k in range(top_k.size(1)):
                                    expert_idx = top_k[b, k].item()
                                    val_expert_tracker[simplified_d_type]['counts'][expert_idx] += 1
                                val_expert_tracker[simplified_d_type]['calls'] += 1
                        # =================================================
                        ## Reconstruct the image and compute the metrics
                        val_batch_size = vx.shape[0]
                        for bt in range(val_batch_size):
                            constructed_hsi = val_dec[bt]
                            GT = vgt[bt]
                            l1loss.append(F.l1_loss(constructed_hsi, GT).item())
                            constructed_hsi = constructed_hsi.cpu().detach().numpy()
                            GT = GT.cpu().detach().numpy()

                            sams.append(sam(constructed_hsi, GT))
                            psnrs.append(psnr(constructed_hsi, GT))
                            rmses.append(rmse(constructed_hsi, GT))
                            ergass.append(ergas(constructed_hsi, GT))

                        # Free the cache every 10 validation samples to avoid fragmentation
                        if ind2 % 10 == 0:
                            torch.cuda.empty_cache()

                    except Exception as e:
                        plog(f"[ERROR] Validation crashed at sample {ind2}, file={vfn if 'vfn' in locals() else 'N/A'}: {type(e).__name__}: {e}", savename)
                        import traceback
                        plog(traceback.format_exc(), savename)
                        raise  # re-raise to keep the full traceback

                # Compute the average time and log it
                ep = time.time() - start_time
                ep = ep / len(sams) if len(sams) > 0 else 0
                torch.cuda.empty_cache()

                # ========== Build the expert utilization report and save it to log ==========
                report_str = f"\n=== Epoch {epoch} Validation Expert Utilization ===\n"
                headers = [f"E{i}" for i in range(10)]
                report_str += f"{'Degradation':<12} | " + " | ".join([f"{h:>5}" for h in headers]) + "\n"
                report_str += "-" * (15 + 8 * 10) + "\n"

                # Sort the entries for more regular log output
                for d_type in sorted(val_expert_tracker.keys()):
                    stats = val_expert_tracker[d_type]
                    if stats['calls'] > 0:
                        freqs = stats['counts'] / stats['calls']
                    else:
                        freqs = stats['counts']
                    act_strs = [f"{f:.3f}" for f in freqs]
                    report_str += f"{d_type:<12} | " + " | ".join([f"{a:>5}" for a in act_strs]) + "\n"
                report_str += "=" * (15 + 8 * 10) + "\n"

                plog(report_str, savename)

                # Prepare the log message
                log_msg = (
                    f"[epoch: {epoch}, batch: {batch_idx + 1}] "
                    f"Phase: {phase}, "
                    f"Total-Loss: {running_loss / len(train_loader):.3f}, "
                )

                if phase == "router_pretrain":
                    # Added: print route-loss separately during router pretraining
                    log_msg += f"route-loss: {running_route_loss / len(train_loader):.4f}, "

                if phase != "router_pretrain":
                    log_msg += (
                        f"L1loss: {running_loss1 / len(train_loader):.3f}, "
                        f"bandMSE: {running_loss2 / len(train_loader):.3f}, "
                        f"sam-loss: {running_loss3 / len(train_loader):.3f}, "
                        f"swt-loss: {running_loss4 / len(train_loader):.3f}, "
                        # Added: print route-loss during joint training
                        f"route-loss: {running_route_loss / len(train_loader):.4f}, "
                        f"val-L1Loss: {np.mean(l1loss) if len(l1loss) > 0 else 0:.3f}, "
                        f"val-RMSE: {np.mean(rmses) if len(rmses) > 0 else 0:.3f}, "
                        f"val-ERGAS: {np.mean(ergass) if len(ergass) > 0 else 0:.3f}, "
                        f"val-SAM: {np.mean(sams) if len(sams) > 0 else 0:.3f}, "
                        f"val-PSNR: {np.mean(psnrs) if len(psnrs) > 0 else 0:.3f}, "
                    )

                log_msg += (
                    f"AVG-Time: {ep:.3f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                )

                plog(log_msg, savename)

                # Log the validation metrics to TensorBoard
                if phase != "router_pretrain":
                    writer.add_scalar("Validation/RMSE", np.mean(rmses), step)
                    writer.add_scalar("Validation/ERGAS", np.mean(ergass), step)
                    writer.add_scalar("Validation/SAM", np.mean(sams), step)
                    writer.add_scalar("Validation/PSNR", np.mean(psnrs), step)

                    writer.add_scalar("Training/L1Loss", running_loss1, step)
                    writer.add_scalar("Training/BandWiseLoss", running_loss2, step)
                    writer.add_scalar("Training/SAMLoss", running_loss3, step)
                    writer.add_scalar("Training/SWTLoss", running_loss4, step)

                writer.add_scalar("Training/Total running loss", running_loss, step)
                writer.add_scalar("Training/RouteLoss", running_route_loss, step)

                # Save the best model
                if phase != "router_pretrain" and best_sam > np.mean(sams) if len(sams) > 0 else math.inf:
                    best_sam = np.mean(sams) if len(sams) > 0 else math.inf
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "sam": np.mean(sams) if len(sams) > 0 else math.inf,
                            "psnr": np.mean(psnrs) if len(psnrs) > 0 else 0,
                            "rmse": np.mean(rmses) if len(rmses) > 0 else 0,
                            "ergas": np.mean(ergass) if len(ergass) > 0 else 0,
                            "epoch": epoch,
                            "lr": scheduler.get_last_lr()[0],
                            'optimizer': optimizer.state_dict(),
                        },
                        f"ckpt/BEST-{savename}.pth",
                    )

        # Save the model every 50 epochs
        if epoch % 50 == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model": model.state_dict(),
                    "sam": np.mean(sams) if len(sams) > 0 else math.inf,
                    "psnr": np.mean(psnrs) if len(psnrs) > 0 else 0,
                    "rmse": np.mean(rmses) if len(rmses) > 0 else 0,
                    "ergas": np.mean(ergass) if len(ergass) > 0 else 0,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    'optimizer': optimizer.state_dict(),
                },
                f"ckpt/EP{epoch}-{savename}.pth",
            )

        model.train()
        scheduler.step()
        step += 1

    ################ Testing #################
    test_loader = torch.utils.data.DataLoader(S3MDHRDataset(
        args.root, img_size=args.imgsize, long_prompt=args.long, mode="test", interpolate=args.intp
    ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # Load the best model
    state_dict = torch.load(f"ckpt/BEST-{savename}.pth")
    model.load_state_dict(state_dict['model'])
    model.to('cuda')
    model.eval()

    rmses, sams, psnrs, ergass = [], [], [], []

    # Start testing
    for data in test_loader:
        x, t, gt, ori = data["source"], data["degra_type"], data["target"], data["ori"]
        x = x.to('cuda').float()
        gt = gt.to('cuda').float()
        ori = ori.to('cuda').float()

        # Encode the text description with the CLIP model
        t_fea = torch.empty(x.shape[0], 1, 512).to('cuda')
        for i in range(x.shape[0]):
            vt_tok = clip.tokenize([t[i]]).to('cuda')
            with torch.no_grad():
                t_fea[i, :, :] = model_clip.encode_text(vt_tok).to('cuda')

        # Forward pass
        y = model(x, ori)

        # Compute the test metrics
        gt = gt[0].cpu().detach().numpy()
        y = y[0].cpu().detach().numpy()
        sams.append(sam(y, gt))
        psnrs.append(psnr(y, gt))
        rmses.append(rmse(y, gt))
        ergass.append(ergas(y, gt))

    # Log the test results
    plog(
        "\n[Testing:]\n test-PSNR: %.3f, test-SAM: %.3f, test-RMSE: %.3f, test-ERGAS: %.3f"
        % (
            np.mean(psnrs),
            np.mean(sams),
            np.mean(rmses),
            np.mean(ergass),
        ), savename
    )


if __name__ == "__main__":
    from options import options as args

    trainer(args)
