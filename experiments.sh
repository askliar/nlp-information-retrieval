epochs=10

#baseline

#python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy &&
python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection RNN1  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy &&
#python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy --negbackprop False &&

#python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy &&
python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection RNN1  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy &&
#python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity easy --negbackprop False &&



#python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard &&
python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection RNN1  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard &&
#python clean_code/main.py  --onlybin True --captions False --augment False --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard --negbackprop False &&

#python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection CBOW  --concat True --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard &&
python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection RNN1  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard &&
#python clean_code/main.py  --onlybin True --captions False --augment True --epochs $epochs --cosine True --projection CBOW  --concat False --sequential False --batch_size 128 --test_batch_size 32 --image_layer None --lr 0.001 --complexity hard --negbackprop False &&



echo 'End'
