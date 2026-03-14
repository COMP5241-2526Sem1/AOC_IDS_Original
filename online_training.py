import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from utils import *
from visualization import plot_training_summary
import time
import shutil
import argparse
import warnings
# 记录程序启动时间
START_TIME = time.time()
# 设定安全退出时间：11.5 小时 (11.5 * 3600 秒)
MAX_RUN_SECONDS = 11.5 * 3600

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='nsl')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--flip_percent", type=float, default=0.2)
parser.add_argument("--sample_interval", type=int, default=2000)
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every X steps")

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
flip_percent = args.flip_percent
sample_interval = args.sample_interval
cuda_num = args.cuda

tem = 0.02
bs = 128
seed = 5009
seed_round = 1

if dataset == 'nsl':
    input_dim = 121
elif dataset == 'unsw':
    input_dim = 196
elif dataset == 'cic':
    input_dim = None
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    # 'labels2' means normal and abnormal, 'labels9' means 'attack_seen', 'attack_unseen', and normal
    # Create an instance of SplitData for 'nsl'
    splitter_nsl = SplitData(dataset='nsl')
    # Transform the data
    x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
    x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')

elif dataset == 'unsw':
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    # Create an instance of SplitData for 'unsw'
    splitter_unsw = SplitData(dataset='unsw')

    # Transform the data
    x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
    x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

else:  # cic
    CICTrain_dataset_path = "/kaggle/input/datasets/chenghinchan/cic-pre-data/CICTrain.csv"
    CICTest_dataset_path  = "/kaggle/input/datasets/chenghinchan/cic-pre-data/CICTest.csv"

    CICTrain = load_data(CICTrain_dataset_path)
    CICTest  = load_data(CICTest_dataset_path)

    splitter_cic = SplitData(dataset='cic')
    x_train, y_train = splitter_cic.transform(CICTrain, labels='label')
    x_test, y_test   = splitter_cic.transform(CICTest,  labels='label')
    input_dim = x_train.shape[1]
    print(f'CIC-IDS-2017 input_dim = {input_dim}')

# Convert to torch tensors
x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")

criterion = CRCLoss(device, tem)

for i in range(seed_round):
    setup_seed(seed+i)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('result', f'{dataset}_seed{seed+i}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    first_round_losses = []
    online_losses = []
    online_metrics = {}

    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent, random_state=seed+i)
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    model = AE(input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
    
# ==================== 断点续跑：加载逻辑 ====================
    start_count = 0
    skip_stage1 = False
    if args.resume and os.path.exists(args.resume):
        print(f"[*] 发现存档，正在从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_count = checkpoint.get('count', 0)
        first_round_losses = checkpoint.get('first_round_losses', [])
        online_losses = checkpoint.get('online_losses', [])
        online_metrics = checkpoint.get('online_metrics', {})
        
        # 恢复 Stage 2 循环所需的核心数据集变量
        x_train_this_epoch = checkpoint['x_train_this_epoch'].to(device)
        x_test_left_epoch = checkpoint['x_test_left_epoch'].to(device)
        y_train_this_epoch = checkpoint['y_train_this_epoch'].to(device)
        y_test_left_labels = checkpoint['y_test_left_labels'].to(device)
        y_train_detection = checkpoint['y_train_detection'].to(device)
        
        skip_stage1 = True
        print(f"[*] 成功恢复！将从 Stage 2 的第 {start_count} 步继续...")
    else:
        first_round_losses = []
        online_losses = []
        online_metrics = {}

####################### Stage 1: Offline Training #######################
    if not skip_stage1: # 如果没有存档，才跑第一阶段
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)

                labels = labels.to(device)
                optimizer.zero_grad()

                features, recon_vec = model(inputs)
                loss = criterion(features,labels) + criterion(recon_vec,labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            first_round_losses.append(avg_loss)
            print(f'[Stage1] seed={seed+i}, epoch={epoch+1}/{epochs}, loss={avg_loss:.6f}')

        x_train = x_train.to(device)
        x_test = x_test.to(device)
        online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

        x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()
        
        y_train_detection = y_train_this_epoch
        y_test_left_labels = y_test_left_epoch.clone()

####################### Stage 2: Online Training #######################
    count = start_count # 从存档点或 0 开始
    
    # 计算正确的总进度条
    if not skip_stage1:
        total_online_samples = len(x_test_left_epoch)
    else:
        total_online_samples = count * sample_interval + len(x_test_left_epoch)
    total_online_steps = (total_online_samples + sample_interval - 1) // sample_interval

    try:
        while len(x_test_left_epoch) > 0:
            # ==================== 12 小时防暴毙检查 ====================
            if time.time() - START_TIME > MAX_RUN_SECONDS:
                
                print(f"\n[!] 警告：运行已达 11.5 小时，为防止 Kaggle 强制终止，执行紧急存档！")
                
                # 1. 保存紧急 Checkpoint
                ckpt_path = os.path.join('result', f'ckpt_timeout_step_{count}.pth')
                torch.save({
                    'count': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'x_train_this_epoch': x_train_this_epoch,
                    'x_test_left_epoch': x_test_left_epoch,
                    'y_train_this_epoch': y_train_this_epoch,
                    'y_test_left_labels': y_test_left_labels,
                    'y_train_detection': y_train_detection,
                    'first_round_losses': first_round_losses,
                    'online_losses': online_losses,
                    'online_metrics': online_metrics
                }, ckpt_path)
                print(f"[*] 超时存档已保存至: {ckpt_path}")
                
                # 2. 自动把整个 result 文件夹打包成 zip，放在 /kaggle/working 根目录
                print("[*] 正在自动打包 result 文件夹...")
                shutil.make_archive('/kaggle/working/AOC_IDS_AutoBackup', 'zip', 'result')
                print("[*] 打包完成！生成文件：/kaggle/working/AOC_IDS_AutoBackup.zip")
                
                # 3. 安全退出循环
                break
            
            count += 1
            processed = min(count * sample_interval, total_online_samples)

            if len(x_test_left_epoch) < sample_interval:
                x_test_this_epoch = x_test_left_epoch.clone()
                y_true_this_step = y_test_left_labels.clone()
                x_test_left_epoch.resize_(0)
                y_test_left_labels.resize_(0)
            else:
                x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
                y_true_this_step = y_test_left_labels[:sample_interval].clone()
                x_test_left_epoch = x_test_left_epoch[sample_interval:]
                y_test_left_labels = y_test_left_labels[sample_interval:]

            with torch.no_grad():
                normal_data = online_x_train[(online_y_train == 0).squeeze()]
                enc, dec = model(normal_data)
                normal_temp = torch.mean(F.normalize(enc, p=2, dim=1), dim=0)
                normal_recon_temp = torch.mean(F.normalize(dec, p=2, dim=1), dim=0)
            predict_label = evaluate(normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection, x_test_this_epoch, 0, model)

            y_true_np = y_true_this_step.cpu().numpy()
            y_pred_np = predict_label.cpu().numpy() if isinstance(predict_label, torch.Tensor) else np.array(predict_label)
            batch_acc  = accuracy_score(y_true_np, y_pred_np)
            batch_prec = precision_score(y_true_np, y_pred_np, zero_division=0)
            batch_rec  = recall_score(y_true_np, y_pred_np, zero_division=0)
            batch_f1   = f1_score(y_true_np, y_pred_np, zero_division=0)
            online_metrics[count] = (batch_acc, batch_prec, batch_rec, batch_f1)

            print(f'[Stage2] seed={seed+i}, step={count}/{total_online_steps} '
                f'({100*processed/total_online_samples:.1f}%) | '
                f'Acc={batch_acc:.4f}  Prec={batch_prec:.4f}  '
                f'Rec={batch_rec:.4f}  F1={batch_f1:.4f}')

            y_test_pred_this_epoch = predict_label
            y_train_detection = torch.cat((y_train_detection.to(device), torch.tensor(y_test_pred_this_epoch).to(device)))
            num_zero = int(flip_percent * y_test_pred_this_epoch.shape[0])
            zero_indices = np.random.choice(y_test_pred_this_epoch.shape[0], num_zero, replace=False)
            y_test_pred_this_epoch[zero_indices] = 1 - y_test_pred_this_epoch[zero_indices]

            x_train_this_epoch = torch.cat((x_train_this_epoch.to(device), x_test_this_epoch.to(device)))
            y_train_this_epoch_temp = y_train_this_epoch.clone()
            y_train_this_epoch = torch.cat((y_train_this_epoch_temp.to(device), torch.tensor(y_test_pred_this_epoch).to(device)))

            train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
            
            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds, batch_size=bs, shuffle=True)
            model.train()
            step_loss = 0.0
            step_batches = 0
            for epoch in range(epoch_1):
                for j, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)

                    labels = labels.to(device)
                    optimizer.zero_grad()

                    features, recon_vec = model(inputs)

                    loss = criterion(features,labels) + criterion(recon_vec,labels)

                    loss.backward()
                    optimizer.step()

                    step_loss += loss.item()
                    step_batches += 1

            online_losses.append(step_loss / max(step_batches, 1))
            # ==================== 周期性自动保存 ====================
            if count % args.save_interval == 0:
                ckpt_path = os.path.join(run_dir, f'ckpt_step_{count}.pth')
                torch.save({
                    'count': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'x_train_this_epoch': x_train_this_epoch,
                    'x_test_left_epoch': x_test_left_epoch,
                    'y_train_this_epoch': y_train_this_epoch,
                    'y_test_left_labels': y_test_left_labels,
                    'y_train_detection': y_train_detection,
                    'first_round_losses': first_round_losses,
                    'online_losses': online_losses,
                    'online_metrics': online_metrics
                }, ckpt_path)
                print(f"[*] 已自动存档至: {ckpt_path}")

    except KeyboardInterrupt:
        # ==================== 捕获中断：紧急死亡保存 ====================
        print(f"\n[!] 检测到手动中断！正在紧急保存当前进度 (Step {count})...")
        ckpt_path = os.path.join('result', f'ckpt_emergency_step_{count}.pth')
        torch.save({
            'count': count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'x_train_this_epoch': x_train_this_epoch,
            'x_test_left_epoch': x_test_left_epoch,
            'y_train_this_epoch': y_train_this_epoch,
            'y_test_left_labels': y_test_left_labels,
            'y_train_detection': y_train_detection,
            'first_round_losses': first_round_losses,
            'online_losses': online_losses,
            'online_metrics': online_metrics
        }, ckpt_path)
        print(f"[*] 紧急存档已保存至: {ckpt_path}")
        print("程序安全退出，下次可以使用 --resume 参数接着跑。")
        exit(0) # 退出程序，避免中断后继续执行最终评估代码报错

################### Final Evaluation ###################
    with torch.no_grad():
        normal_data = online_x_train[(online_y_train == 0).squeeze()]
        enc, dec = model(normal_data)
        normal_temp = torch.mean(F.normalize(enc, p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(F.normalize(dec, p=2, dim=1), dim=0)

    res_en, res_de, res_final, y_pred_final = evaluate(
        normal_temp, normal_recon_temp, x_train_this_epoch, y_train_detection,
        x_test, y_test, model, return_predictions=True)

    print(f'\n{"=" * 60}')
    print(f'  Final Results - {dataset.upper()} seed={seed+i}')
    print(f'{"=" * 60}')
    print(f'  {"":12s} {"Acc":>8s} {"Prec":>8s} {"Recall":>8s} {"F1":>8s}')
    print(f'  {"-" * 44}')
    print(f'  {"Encoder":12s} {res_en[0]:>8.4f} {res_en[1]:>8.4f} {res_en[2]:>8.4f} {res_en[3]:>8.4f}')
    print(f'  {"Decoder":12s} {res_de[0]:>8.4f} {res_de[1]:>8.4f} {res_de[2]:>8.4f} {res_de[3]:>8.4f}')
    print(f'  {"Combined":12s} {res_final[0]:>8.4f} {res_final[1]:>8.4f} {res_final[2]:>8.4f} {res_final[3]:>8.4f}')
    print(f'{"=" * 60}')

    # ==================== Save results to result/ ====================
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    run_result = {
        'config': {
            'dataset': dataset,
            'seed': seed + i,
            'epochs': epochs,
            'epoch_1': epoch_1,
            'percent': percent,
            'flip_percent': flip_percent,
            'sample_interval': sample_interval,
            'batch_size': bs,
            'temperature': tem,
            'input_dim': input_dim,
            'num_first_train': num_of_first_train,
            'timestamp': timestamp,
        },
        'stage1_losses': first_round_losses,
        'stage2_losses': online_losses,
        'online_metrics': {
            str(step): dict(zip(metric_names, [float(v) for v in vals]))
            for step, vals in online_metrics.items()
        },
        'final_results': {
            'encoder':  dict(zip(metric_names, [float(v) for v in res_en])),
            'decoder':  dict(zip(metric_names, [float(v) for v in res_de])),
            'combined': dict(zip(metric_names, [float(v) for v in res_final])),
        },
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(run_result, f, indent=2)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(run_dir, 'model.pth'))

    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)
    np.savez(
        os.path.join(run_dir, 'predictions.npz'),
        y_true=y_test_np,
        y_pred=y_pred_final,
    )

    plot_training_summary(
        first_round_losses=first_round_losses,
        online_losses=online_losses,
        online_metrics=online_metrics,
        final_encoder=res_en,
        final_decoder=res_de,
        final_combined=res_final,
        y_test_true=y_test_np,
        y_test_pred=y_pred_final,
        dataset=dataset,
        seed=seed+i,
        save_dir=run_dir,
    )

    print(f'  [Saved] All results -> {run_dir}/')
