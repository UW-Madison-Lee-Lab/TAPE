save_path="./retriever/database/wikipedia"
mkdir -p $save_path
python retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
rm $save_path/part_*
gzip -d $save_path/wiki-18.jsonl.gz
rm $save_path/wiki-18.jsonl.gz