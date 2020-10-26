export CUDA_VISIBLE_DEVICES=2
export DATA_DIR='ko_test_data/source_data'
export KEY_DIR='data/LDA_keyword'
export TOPIC='test'
export PRED_DIR='result'

python pplm_mass.py ./ko_test_data/processed_data/economy_processed/ \
	--path MASS_model/Ko_MASS_Summarization_fine_tuning_epoch_2.pt \
	--user-dir mass \
	--task translation_mass \
	--batch-size 4 \
	--beam 5 \
	--min-len 15 \
	--no-repeat-ngram-size 3 \
	--lenpen 1.0 \
	--data_dir $DATA_DIR/$TOPIC.src \
	--bag_of_words $KEY_DIR/test_200.key \
	--output_dir $PRED_DIR \
	--top_k 50 \
	--step_size 0.03 \
	--num_samples 1 \
	--gamma	1.5 \
	--gm_scale 0.95 \
	--kl_scale 0.01 \
	--temperature_pplm 0.9 \
	--num_iterations 1 \
	--window_length 3 \
