def CreateDataset(opt, train):
    if opt.dataset in ['shapenet']:
        from datasets.shapenet_data_loader import ShapeNetDataset
        dataset = ShapeNetDataset(opt, train)
    elif opt.dataset in ['kitti']:
        from datasets.kitti_data_loader import KITTIDataset
        dataset = KITTIDataset(opt, train)
        print("dataset [%s] was created" % (dataset.name()))
    return dataset