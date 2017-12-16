epochs=1
#baseline
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&
python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&

#vary concat
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2   &&

python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2   &&

#same for mse
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&
python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&

python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2   &&

python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2   &&


# hard difficulty
#baseline
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 --complexity hard &&
python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 --complexity hard &&

#vary concat
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 --complexity hard &&

python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2  --complexity hard &&

#same for mse
python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 --complexity hard &&
python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 --complexity hard &&

python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2  --complexity hard &&

python clean_code/main.py  --onlybin False --captions True --augment True --epochs $epochs --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2  --complexity hard &&


# R U NAN

python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection RNN1  --concat True --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&

python clean_code/main.py  --onlybin False --captions True --augment False --epochs $epochs --cosine True --projection RNN1  --concat False --sequential False --batch_size 1024 --test_batch_size 32 --image_layer mlp2 &&
echo 'End'
