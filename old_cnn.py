from final_cnn import final_cnn_loop

group = ['Cut', 'Cut+flip']
data_dir = ['./cut_to_4/3_split_test', './cut_to_4_flip/3_split']
predict_dir = ['./cut_to_4/3_split', './cut_to_4_flip/3_split_test']

for i in range(len(group)):
	group = group[i] 
	data_dir = data_dir[i]
	predict_ds = predict_dir[i]
	final_cnn_loop(group , data_dir, predict_dir)