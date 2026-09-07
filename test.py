import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clip
import os
from utils.dataset import TestDataset
from models.SMDHR import SMDHR_b


from tqdm import tqdm
from utils.lossfunc import HyperspectralSWTLoss, SAMLoss, BandWiseMSE
from utils.metrics import psnr, sam, rmse, ergas
import time
import torch.nn.functional as F
import cv2

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def write_img(filename, img, to_uint=True):
    if to_uint: img = np.round(img * 255.0).astype('uint8')
    cv2.imwrite(filename, img[:, :, ::-1])


def plot_mean_spectral_diff1(spectral_diff_dict, restored_path):
    """
    Plot the mean spectral difference curve for each degradation type and save to Excel.

    Args:
        spectral_diff_dict (dict): stores the per-band differences of each degradation type
        restored_path (str): path where the curve images are saved
    """
    for degra_type, diff_list in spectral_diff_dict.items():
        if diff_list:  # ensure the list is not empty
            print(degra_type)
            # compute the mean band difference of all images
            mean_spectral_diff = np.mean(diff_list, axis=0)
            print(mean_spectral_diff.shape)
            # plot the curve
            plt.figure()
            plt.plot(mean_spectral_diff)
            plt.title(f'{degra_type} Mean spectral difference curve of hyperspectral testing images')
            plt.xlabel('Band Number')
            plt.ylabel('Mean Different')
            plt.grid(True)
            # save the curve image
            plt.savefig(os.path.join(restored_path, f'{degra_type}_spectral_diff.png'))
            plt.close()


def plot_mean_spectral_diff(spectral_diff_dict, restored_path):
    """
    Plot the mean spectral difference curve for each degradation type and save to Excel (all results in one sheet).

    Args:
        spectral_diff_dict (dict): stores the per-band differences of each degradation type
        restored_path (str): path where the curve images are saved
    """
    # Create a DataFrame to store the mean spectral difference of all degradation types
    df_all = pd.DataFrame()

    for degra_type, diff_list in spectral_diff_dict.items():
        if diff_list:  # ensure the list is not empty
            # compute the mean band difference of all images
            mean_spectral_diff = np.mean(diff_list, axis=0)

            # plot the curve
            plt.figure()
            plt.plot(mean_spectral_diff)
            plt.title(f'{degra_type} Mean spectral difference')
            plt.xlabel('Band Number')
            plt.ylabel('Mean Different')
            plt.grid(True)
            plt.xlim(1, 305)
            plt.xticks(np.arange(1, 306, 50))
            # save the curve image
            plt.savefig(os.path.join(restored_path, f'{degra_type}_spectral_diff.png'))
            plt.close()

            # Add the mean spectral difference of the current degradation type to the DataFrame
            if df_all.empty:
                df_all['band'] = range(1, len(mean_spectral_diff) + 1)
            df_all[degra_type] = mean_spectral_diff

    # Save to Excel (all results in one sheet)
    excel_path = os.path.join(restored_path, 'mean_spectral_diff.xlsx')
    df_all.to_excel(excel_path, sheet_name='Mean Spectral Difference', index=False)


def plot_total_mean_spectral_diff(spectral_diff_dict, restored_path):
    """
    Plot the mean spectral difference curves of all degradation types in a single chart.

    Args:
        spectral_diff_dict (dict): stores the per-band differences of each degradation type
        restored_path (str): path where the curve image is saved
    """
    plt.figure()
    avg_spectral_diff = []
    for degra_type, diff_list in spectral_diff_dict.items():
        if diff_list:  # ensure the list is not empty
            mean_spectral_diff = np.mean(diff_list, axis=0)
            avg_spectral_diff.append(mean_spectral_diff)
            plt.plot(mean_spectral_diff, label=degra_type)

    meanavg_spectral_diff = np.mean(avg_spectral_diff, axis=0)
    plt.plot(meanavg_spectral_diff, label='average')
    plt.title('Total Mean spectral difference')
    plt.xlabel('Band Number')
    plt.ylabel('Mean Different')
    plt.grid(True)
    plt.legend()
    plt.xlim(1, 305)
    plt.xticks(np.arange(1, 306, 50))
    # save the curve image
    plt.savefig(os.path.join(restored_path, 'total_spectral_diff.png'))
    plt.close()

def plot_avg_mean_spectral_diff(spectral_diff_dict, restored_path):
    """
    Plot the mean spectral difference curves of all degradation types in a single chart.

    Args:
        spectral_diff_dict (dict): stores the per-band differences of each degradation type
        restored_path (str): path where the curve image is saved
    """
    plt.figure()
    avg_spectral_diff = []
    for degra_type, diff_list in spectral_diff_dict.items():
        if diff_list:  # ensure the list is not empty
            mean_spectral_diff = np.mean(diff_list, axis=0)
            avg_spectral_diff.append(mean_spectral_diff)

    meanavg_spectral_diff = np.mean(avg_spectral_diff, axis=0)
    plt.plot(meanavg_spectral_diff)
    plt.title('Total Mean spectral difference')
    plt.xlabel('Band Number')
    plt.ylabel('Mean Different')
    plt.grid(True)
    plt.legend()
    plt.xlim(1, 305)
    plt.xticks(np.arange(1, 306, 50))
    # save the curve image
    plt.savefig(os.path.join(restored_path, 'avg_spectral_diff.png'))
    plt.close()

def test(args):
    img_size = (args.imgsize, args.imgsize)
    win_size = tuple(args.win_size)
    patch_size = tuple(args.pat_size)
    n_layers = tuple(args.n_layers)

    model = eval(args.savename)()
    print(args.savename,args.ckpt)

    # Load the state dict
    state_dict = torch.load(args.ckpt)['model']
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    print("Loaded the state dict")

    restored_path = os.path.join('restored_results/GF5', args.savename)
    if not os.path.exists(restored_path):
        os.makedirs(restored_path, exist_ok=True)
    # Create an Excel record file (using pandas)
    excel_path = os.path.join(restored_path, 'metrics_results.xlsx')
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=['degra_type', 'SAM', 'PSNR', 'RMSE', 'ERGAS', 'Time'])
        df.to_excel(excel_path, index=False)

    degraded_types = ['h', 'b', 'n', 'bm',
                      'h_b', 'h_n', 'h_bm', 'b_n', 'b_bm', 'n_bm',
                      'h_b_n', 'h_b_bm', 'h_n_bm', 'b_n_bm', 'h_b_n_bm']
    # Initialize the dict that stores per-band differences
    spectral_diff_dict = {degra_type: [] for degra_type in degraded_types}
    for i in range(len(degraded_types)):
        SAM = AverageMeter()
        PSNR = AverageMeter()
        RMSE = AverageMeter()
        ERGAS = AverageMeter()
        TIME = AverageMeter()

        degraded_type = degraded_types[i]
        dataloader = torch.utils.data.DataLoader(
            TestDataset(args.root, img_size=256, long_prompt=False, interpolate=True, mode='test', degra_type=degraded_type),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

        with torch.no_grad():
            model_clip, _ = clip.load("ViT-B/32", device="cpu")
            model_clip.eval()

        batch_idx = 0
        test_bar = tqdm(dataloader)
        if not os.path.exists(os.path.join(restored_path, degraded_type)):
            os.makedirs(os.path.join(restored_path, degraded_type), exist_ok=True)
        f_result = open(os.path.join(restored_path, degraded_type, 'results.csv'), 'w')

        print("Start generating testing results...")
        # Ensure all CUDA operations are complete (only when using GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


        for step, (data) in enumerate(test_bar):
            with torch.no_grad():
                x, gt, t, fn, l, ori = data["source"], data["target"], data["degra_type"], data["filename"][0], data["label"], data["ori"]
                x = x.to('cuda').float()
                gt = gt.to('cuda').float()
                ori = ori.to('cuda').float()
                t_fea = torch.empty(x.shape[0], 512).to('cuda')
                l= l.to('cuda').float()
                H, W = x.shape[2:]
                for i in range(x.shape[0]):
                    vt_tok = clip.tokenize([t[i]])
                    with torch.no_grad():
                        t_fea[i, :] = model_clip.encode_text(vt_tok).to('cuda')
                x = pad_img(x, model.patch_size if hasattr(model, 'patch_size') else 16)#56
                gt = pad_img(gt, model.patch_size if hasattr(model, 'patch_size') else 16)#56
                ori = pad_img(ori, model.patch_size if hasattr(model, 'patch_size') else 16)
                # Ensure all CUDA operations are done
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_event = time.perf_counter()  # use a high-precision timer
                y = model(x,ori)
                end_event = time.perf_counter()
                y = y[:, :, :H, :W]
                time_c = end_event - start_event  # convert to seconds
                constructed_hsi = y.cpu().detach().numpy()
                GT = gt[:, :, :H, :W].cpu().detach().numpy()
                sam_val=sam(constructed_hsi, GT)
                psnr_val=psnr(constructed_hsi, GT)
                rmse_val=rmse(constructed_hsi, GT)
                ergas_val=ergas(constructed_hsi, GT)
                # Compute the mean per-band difference
                spectral_diff = np.mean(np.abs(constructed_hsi[0] - GT[0]), axis=(1, 2))  # (C,)
                spectral_diff_dict[degraded_type].append(spectral_diff)

                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()  # (C, H, W)
                gt = gt[0].cpu().detach().numpy()
                x_rgb = np.stack((x[58, :, :], x[39, :, :], x[20, :, :]))
                y_rgb = np.stack((y[58, :, :], y[39, :, :], y[20, :, :]))
                gt_rgb = np.stack((gt[58, :, :], gt[39, :, :], gt[20, :, :]))
                x_rgb = x_rgb.transpose(1, 2, 0)
                y_rgb = y_rgb.transpose(1, 2, 0)  # (H, W, C)
                gt_rgb = gt_rgb.transpose(1, 2, 0)
                if not os.path.exists(os.path.join(restored_path, degraded_type, 'imgs')):
                    os.makedirs(os.path.join(restored_path, degraded_type, 'imgs'), exist_ok=True)

                SAM.update(sam_val)
                PSNR.update(psnr_val)
                RMSE.update(rmse_val)
                ERGAS.update(ergas_val)  # update SAM
                TIME.update(time_c)

                # Print progress
                print(f'Test: [{step}]\t'
                      f'SAM: {SAM.val:.03f} ({SAM.avg:.03f})\t'
                      f'PSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})\t'
                      f'RMSE: {RMSE.val:.03f} ({RMSE.avg:.03f})\t'
                      f'ERGAS: {ERGAS.val:.03f} ({ERGAS.avg:.03f})\t'
                      f'TIME: {TIME.val:.03f} ({TIME.avg:.03f})')
                f_result.write('%s,%.03f,%.02f,%.03f,%.03f\n' % (fn, sam_val, psnr_val, rmse_val, ergas_val))


        f_result.close()
        # Save the final results to Excel
        new_row = {
            'degra_type': degraded_type,
            'SAM': SAM.avg,
            'PSNR': PSNR.avg,
            'RMSE': RMSE.avg,
            'ERGAS': ERGAS.avg,
            'Time': TIME.avg
        }
        # Read the existing data and append the new result
        df_existing = pd.read_excel(excel_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        df_updated.to_excel(excel_path, index=False)

        print(f"最终结果已保存到：{excel_path}")
        print(f"指标汇总：{new_row}")

    # Call the function to plot and save the mean spectral difference curves and the Excel file
    plot_mean_spectral_diff(spectral_diff_dict, restored_path)
    # Call the function to plot and save the overall mean spectral difference curve
    plot_total_mean_spectral_diff(spectral_diff_dict, restored_path)
    plot_avg_mean_spectral_diff(spectral_diff_dict, restored_path)

    print("Finish generating testing results...")

if __name__ == "__main__":
    from options import options as args
    
    test(args)