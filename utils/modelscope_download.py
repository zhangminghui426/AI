#模型下载
from modelscope import snapshot_download

if __name__ == "__main__":
    ms = 'qwen/Qwen2-7B-Instruct'
    save_dir = "/root/autodl-tmp/work/model/Qwen/Qwen2-7B-Instruct"
    snapshot_download(ms, cache_dir=save_dir)
