import torch


def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def get_category(num):
    if num == '0':
        return 'Crater'
    if num == '2':
        return 'Dark Dune'
    if num == '15':
        return 'Slope Streak'
    if num == '4':
        return 'Bright Dune'
    if num == '5':
        return 'Impact Ejecta'
    if num == '7':
        return 'Spider'
    return None

def get_voc_bbox(bb, w, h):
    voc = []
    bbox_width = float(bb[2]) * w
    bbox_height = float(bb[3]) * h
    center_x = float(bb[1]) * w
    center_y = float(bb[0]) * h
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))
    return voc


def get_img2bbox(bbox_filenames):
    img2bbox = {}
    for filename in bbox_filenames:
        with open('images/' + filename) as f:
            img_name = filename.split('.')[0] + '.jpg'
            img2bbox[img_name] = [[], []]

            lines = f.readlines()
            for line in lines:
                line_input = line.strip().split(' ')
                bbox = get_voc_bbox(list(map(float, line_input[1:])), w=227, h=227)
                category = get_category(line_input[0])
                img2bbox[img_name][0].append(bbox)
                img2bbox[img_name][1].append(category)
    return img2bbox

