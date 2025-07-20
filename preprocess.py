import argparse
import torch
from transformers import GPT2Model, GPT2Tokenizer
from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess for GPT-2')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id, -1 for CPU')
    parser.add_argument('--llm_ckp_dir', type=str, default='./GPT2', help='GPT-2 checkpoints dir')
    parser.add_argument('--dataset', type=str, default='OBS_FD01_cleaned',
                        help='dataset to preprocess, e.g., OBS_FD01_cleaned')
    args = parser.parse_args()
    print(f"Processing dataset: {args.dataset}")

    # 加载 GPT-2 模型和分词器
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.llm_ckp_dir)
    model = GPT2Model.from_pretrained(args.llm_ckp_dir).to(device)
    model.eval()

    # 设置数据集参数
    seq_len = 672
    label_len = 576
    pred_len = 5

    # 加载自定义数据集
    data_set = Dataset_Preprocess(
        root_path='./dataset/',
        data_path=f'{args.dataset}.csv',
        size=[seq_len, label_len, pred_len],
        time_col='dtime')

    data_loader = DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
    )

    from tqdm import tqdm

    print(f"Total timestamps: {len(data_set.data_stamp)}")
    print(f"Total length: {data_set.tot_len}")
    save_dir_path = './dataset/'
    output_list = []


    # 定义一个函数来处理时间戳嵌入
    def process_batch(data):
        time_stamps = data  # 直接使用 data，因为 Dataset_Preprocess 返回单个时间戳
        batch_embeddings = []
        for stamp in time_stamps:
            inputs = tokenizer(stamp, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            batch_embeddings.append(embedding)
        return torch.stack(batch_embeddings)


    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        output = process_batch(data)
        output_list.append(output.detach().cpu())

    result = torch.cat(output_list, dim=0)
    print(f"Embedding shape: {result.shape}")
    torch.save(result, save_dir_path + f'/{args.dataset}.pt')
    print(f"Embeddings saved to {save_dir_path}/{args.dataset}.pt")