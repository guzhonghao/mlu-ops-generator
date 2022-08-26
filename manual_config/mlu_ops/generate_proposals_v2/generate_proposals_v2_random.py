import sys
import os 
import numpy as np 
def dShape(shapes):
    shape_val = '"shape":['
    for i in range(len(shapes)-1):
        shape_val += str(shapes[i])+','
    shape_val += str(shapes[len(shapes)-1]) + ']'
    return  shape_val

def dType(data_type):
    return '"dtype":"' + data_type + '"'
def dRandomDistribution(start, end):
    return '"random_distribution":{"uniform":[' + str(start) + ',' + str(end) + ']}'
def dlayout(data_layout):
    return '"layout":"' + data_layout + '"'
def genSingleCase(dtype='float32', params_list=[1,1,1,1,1,1,1,1,1]):
    N = params_list[0]
    A = params_list[1]
    H = params_list[2]
    W = params_list[3]

    post_nms_top_n = params_list[4]
    pre_nms_top_n = params_list[5]
    nms_thresh = params_list[6]
    min_size = params_list[7]
    eta = params_list[8]
    pixel_offset = params_list[9]

    scores_shape = [N, A, H, W]
    deltas_shape = [N, 4 * A, H, W]
    anchors_shape = [A, H, W, 4]
    variances_shape = [A, H, W, 4]
    img_shape = [N, 2]

    rois_shape = [pre_nms_top_n, 4]
    rpn_roi_probs_shape = [pre_nms_top_n, 1]
    rpn_rois_num_shape = [N, 1]
    rpn_rois_batch_size_shape = [1]

    bottom_limit = 10
    up_limit = 100

    inputs = '    {\n       "inputs":[\n'
    scores_input = '            {' + dShape(scores_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    deltas_input = '            {' + dShape(deltas_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    anchors_input = '            {' + dShape(anchors_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    variances_input = '            {' + dShape(variances_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    img_input = '            {' + dShape(img_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '}\n'
    

    outputs = '       "outputs":[\n'
    rois_output = '            {' + dShape(rois_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '},\n'
    rpn_roi_probs_output = '            {' + dShape(rpn_roi_probs_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '},\n'
    
    rrpn_rois_num_output = '            {' + dShape(rpn_rois_num_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '},\n'
    rpn_rois_batch_size_output = '            {' + dShape(rpn_rois_batch_size_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '}\n'

    inputs = inputs + scores_input + deltas_input + anchors_input + variances_input +img_input + '            ],\n'
    outputs = outputs + rois_output + rpn_roi_probs_output + rrpn_rois_num_output + rpn_rois_batch_size_output + '            ],\n'
    
    proto_param = '       "op_params":{\n'
    pre_nms_top_n_op =  '            "pre_nms_top_n":' + str(pre_nms_top_n) + ',\n'
    post_nms_top_n_op =  '            "post_nms_top_n":' + str(post_nms_top_n) + ',\n'
    nms_thresh_op =  '            "nms_thresh":' + str(nms_thresh) + ',\n'
    min_size_op =  '            "min_size":' + str(min_size) + ',\n'
    eta_op =  '            "eta":' + str(eta) + ',\n'
    pixel_offset_op =  '            "pixel_offset": true \n'
    if(pixel_offset == 0):
        pixel_offset_op =  '            "pixel_offset": false \n'

    proto_param = proto_param + pre_nms_top_n_op + post_nms_top_n_op + nms_thresh_op + min_size_op + eta_op +  pixel_offset_op + '        }\n'

    cur_res = inputs + outputs + proto_param + '    }'
    return cur_res

def genCase():
    cur_res = '    "manual_data":[\n'

    random = 1
    if random == 1:
        N = np.random.randint(1,5)
        A = np.random.randint(1,50)
        H = np.random.randint(1,50)
        W = np.random.randint(1,50)

        post_nms_top_n = np.random.randint(1,20000)
        pre_nms_top_n = np.random.randint(1,2000)
        nms_thresh = np.random.randint(1,100)/100
        min_size = np.random.randint(1,50)/100
        eta = 1.0
        pixel_offset = np.random.randint(0,10) > 5

        param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
        cur_res += genSingleCase(params_list=param)

        for i in range(80):
            N = 1
            A = np.random.randint(1,50)
            H = np.random.randint(1,50)
            W = np.random.randint(1,50)

            post_nms_top_n = np.random.randint(1,20000)
            pre_nms_top_n = np.random.randint(1,2000)
            nms_thresh = np.random.randint(1,100)/100
            min_size = np.random.randint(1,50)/100
            eta = 1.0
            pixel_offset = np.random.randint(0,10) > 5

            param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
            cur_res += ',\n' + genSingleCase(params_list=param)

            if i % 2 == 0:
                N = np.random.randint(1,5)
                A = np.random.randint(1,100)
                H = np.random.randint(1,50)
                W = np.random.randint(1,50)

                post_nms_top_n = np.random.randint(1,20000)
                pre_nms_top_n = np.random.randint(1,2000)
                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5
                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)


            if i % 3 == 0:
                N = np.random.randint(1,5)
                A = np.random.randint(1,50)
                H = np.random.randint(1,100)
                W = np.random.randint(1,50)

                post_nms_top_n = np.random.randint(1,20000)
                pre_nms_top_n = np.random.randint(1,2000)
                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5
                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)

                
            if i % 5 == 0:
                N = np.random.randint(1,5)
                A = np.random.randint(1,50)
                H = np.random.randint(1,50)
                W = np.random.randint(1,100)

                post_nms_top_n = np.random.randint(1,20000)
                pre_nms_top_n = np.random.randint(1,2000)
                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5
                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)

    else:
        N = 1
        A = 15
        H = 54
        W = 40

        post_nms_top_n = 12000
        pre_nms_top_n = 2000
        nms_thresh = 0.5
        min_size = 0.1
        eta = 1.0
        pixel_offset = 1
        param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
        cur_res += genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 48
        W = 48
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 42
        W = 56
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 40
        W = 61
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 44
        W = 59
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 59
        W = 44
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 42
        W = 64
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 15
        H = 48
        W = 64
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 3
        H = 50
        W = 68
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 3
        H = 160
        W = 216
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 3
        H = 50
        W = 76
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 3
        H = 160
        W = 248
        cur_res += ',\n' + genSingleCase(params_list=param)

        N = 1
        A = 3
        H = 176
        W = 240
        cur_res += ',\n' + genSingleCase(params_list=param)
        # line 15

    cur_res += '\n    ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"generate_proposals_v2",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2","diff3"],\n\
    "if_dynamic_threshold": true,\n'
    res += genCase()
    file = open("./generate_proposals_v2_random.json",'w')
    file.write(res)
    file.close()
 