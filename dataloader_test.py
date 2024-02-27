from dataloader import dataset_wrapper, run_training_step
dataset = dataset_wrapper()

num_iterations = 2

for _ in range(num_iterations):
    try:
        cutouts, catalog, tile = dataset.__next__()  # pull out the an item for training
        run_training_step((cutouts, catalog, tile))
        del cutouts  # cleanup
        del catalog
        del tile
        print('loaded')

    except KeyboardInterrupt:
        break
 
# note that it is likely worth adding something that will wipe the  
# download directory once program is done (or when new one starts)