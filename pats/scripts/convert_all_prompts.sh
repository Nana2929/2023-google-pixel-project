

ROOT="./"
DATA_DIR="$ROOT/processed"
OUTPUT_DIR="$ROOT/converted"

INF_FILE_PREFIX="PATS_for_inference"
EVAL_FILE_PREFIX="PATS_for_eval"

for prompt in "AS" "ASQP" "ASTE" "FREEFORM"; do
    echo "Converting prompt $prompt"
    python src/convert.py   --prompt_format=$prompt \
                                                        --data_dir=$DATA_DIR \
                                                        --output_dir=$OUTPUT_DIR \
                                                        --inf_file_prefix=$INF_FILE_PREFIX \
                                                        --eval_file_prefix=$EVAL_FILE_PREFIX

done

for version in {1..2} ; do
    echo "Converting prompt template alsc"
    echo "Converting version $version"
    python src/alsc_convert.py   --version=$version \
                                                            --data_dir=$DATA_DIR \
                                                            --output_dir=$OUTPUT_DIR \
                                                            --inf_file_prefix=$INF_FILE_PREFIX \
                                                            --eval_file_prefix=$EVAL_FILE_PREFIX
done