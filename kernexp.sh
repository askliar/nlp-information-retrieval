epochs=1
CUDA_LAUNCH_BLOCKING=1 kernprof -l clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 256 --test_batch_size 4 --image_layer mlp2
