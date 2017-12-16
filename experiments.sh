python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine True --projection CBOW  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine True --projection RNN1  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
#vary augment
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine True --projection CBOW  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine True --projection RNN1  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2

#vary concat
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine True --projection CBOW  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine True --projection RNN1  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine True --projection CBOW  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine True --projection RNN1  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2

# change to euclidean

python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine False --projection CBOW  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine False --projection RNN1  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
#vary augment
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine False --projection CBOW  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine False --projection RNN1  --concat True --sequential False --batch_size 256 --test_batch_size 1 --image_layer mlp2

#vary concat
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine False --projection CBOW  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 1 --cosine False --projection RNN1  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine False --projection CBOW  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 1 --cosine False --projection RNN1  --concat False --sequential False --batch_size 256 --test_batch_size 32 --image_layer mlp2




