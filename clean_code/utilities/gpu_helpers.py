from clean_code.utilities.config import Config

config = Config()

if config.CUDA:
    try:
        import gpustat
    except ImportError:
        raise ImportError("pip install gpustat")

    def show_memusage(device=0):
        gpu_stats = gpustat.GPUStatCollection.new_query()
        item = gpu_stats.jsonify()["gpus"][device]
        print("{}/{}".format(item["memory.used"], item["memory.total"]))