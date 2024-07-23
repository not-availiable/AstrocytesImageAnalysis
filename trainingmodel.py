from cellpose import io, models, train
io.logger_setup()

output = io.load_train_test_data(train_dir, test_dir, image_filter="_img",
                                mask_filter="_masks", look_one_level_down=False)
images, labels, image_names, test_images, test_labels, image_names_test = output

# e.g. retrain a Cellpose model
model = models.CellposeModel(model_type="nuclei")

model_path = train.train_seg(model.net, train_data=images, train_labels=labels,
                            channels=[1,2], normalize=True,
                            test_data=test_images, test_labels=test_labels,
                            weight_decay=1e-4, SGD=True, learning_rate=0.1,
                            n_epochs=100, model_name="my_new_model")