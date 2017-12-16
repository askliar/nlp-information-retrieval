#python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32 &&

#vary augment
#python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine True --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32  &&

#vary concat
#python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32   &&

#python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine True --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32   &&

# change to euclidean
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32   &&

#vary augment
python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine False --projection CBOW  --concat True --sequential False --batch_size 1024 --test_batch_size 32   && 


#vary concat
python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32   && 

python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine False --projection CBOW  --concat False --sequential False --batch_size 1024 --test_batch_size 32   && 


#rnns


#python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine True --projection RNN1  --concat True --sequential False --batch_size 1024 --test_batch_size 32 &&

#python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine True --projection RNN1  --concat True --sequential False --batch_size 1024 --test_batch_size 32 &&





#eucl

python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine False --projection RNN1  --concat True --sequential False --batch_size 1024 --test_batch_size 32   &&

python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine False --projection RNN1  --concat True --sequential False --batch_size 1024 --test_batch_size 32   && 

python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine False --projection RNN1  --concat False --sequential False --batch_size 1024 --test_batch_size 32   && 

python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine False --projection RNN1  --concat False --sequential False --batch_size 1024 --test_batch_size 32 && 


#long boi
#python clean_code/main.py  --onlybin False --captions False --augment False --epochs 50 --cosine True --projection RNN1  --concat False --sequential False --batch_size 512 --test_batch_size 32  &&

#python clean_code/main.py  --onlybin False --captions False --augment True --epochs 50 --cosine True --projection RNN1  --concat False --sequential False --batch_size 256 --test_batch_size 32  
