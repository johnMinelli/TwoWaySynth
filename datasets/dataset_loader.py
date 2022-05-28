def CreateDataset(opt, train=False, valid=False, eval=False):
    if opt.dataset in ['shapenet']:
        from datasets.shapenet_data_loader import ShapeNetDataset
        dataset = ShapeNetDataset(opt, train, valid, eval)
    elif opt.dataset in ['kitti']:
        from datasets.kitti_data_loader import KITTIDataset
        dataset = KITTIDataset(opt, train, valid, eval)
        print("dataset [%s] was created" % (dataset.name()))
    return dataset