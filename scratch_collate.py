def multitask_collate_fn(batch):
    """
    Custom collate_fn for heterogeneous batching with bounding boxes.
    Expects batch elements to be (tensor, label_idx, bboxes, class_labels)
    """
    images = []
    labels = []
    bboxes = []
    class_labels = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        bboxes.append(item[2])
        class_labels.append(item[3])
        
    import torch
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, bboxes, class_labels
