import torch

joint_names = [
    'Hip', 'RHip', 'RKnee','RFoot','LHip','LKnee','LFoot',
    'Spine','Thorax','Neck','Head',
    'LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist'
]
edge_names = [
    'RHip', 'RKnee','RFoot','LHip','LKnee','LFoot',
    'Spine','Thorax','Neck','Head',
    'LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist'
]
edge_parent = {
    'RHip': [],
    'RKnee': ['RHip'],
    'RFoot': ['RKnee'],
    'LHip': [],
    'LKnee': ['LHip'],
    'LFoot': ['LKnee'],
    'Spine': [],
    'Thorax': ['Spine'],
    'Neck': ['Thorax'],
    'Head': ['Neck'],
    'LShoulder': ['Thorax'],
    'LElbow': ['LShoulder'],  
    'LWrist': ['LElbow'],
    'RShoulder': ['Thorax'],
    'RElbow': ['RShoulder'],  
    'RWrist': ['RElbow']
}
edge_children = {
    'RHip': ['RKnee'],
    'RKnee': ['RFoot'],
    'RFoot': [],
    'LHip': ['LKnee'],
    'LKnee': ['LFoot'],
    'LFoot': [],
    'Spine': ['Thorax'],
    'Thorax': ['Neck', 'LShoulder', 'RShoulder'],
    'Neck': ['Head'],
    'Head': [],
    'LShoulder': ['LElbow'],
    'LElbow': ['LWrist'],  
    'LWrist': [],
    'RShoulder': ['RElbow'],
    'RElbow': ['RWrist'],  
    'RWrist': []
}
edge_crosshead ={
    'RHip': ['LHip','Spine'],
    'LHip': ['RHip','Spine'],
    'Spine': ['LHip', 'RHip'],
    'LShoulder': ['Neck', 'RShoulder'],
    'RShoulder': ['Neck', 'LShoulder']		
}
edge_index = torch.tensor([
    [1,0], [2,1], [3,2], [4,0], [5,4], [6,5],
    [7,0], [8,7], [9,8], [10,9],
    [11,8], [12,11], [13,12], [14,8], [15,14], [16,15]
])
joint_to_edge_mapping = {
    'Hip': ['RHip', 'LHip', 'Spine'],
    'RHip': ['RHip', 'RKnee' ],
    'RKnee': ['RKnee', 'RFoot'],
    'RFoot': ['RFoot'],
    'LHip': ['LHip', 'LKnee' ],
    'LKnee': ['LKnee', 'LFoot'],
    'LFoot': ['LFoot'],
    'Spine': ['Spine', 'Thorax'],
    'Thorax': ['Thorax', 'Neck', 'LShoulder', 'RShoulder'],
    'Neck': ['Neck', 'Head'],
    'Head': ['Neck'],
    'LShoulder': ['LShoulder', 'LElbow'],
    'LElbow': ['LElbow', 'LWrist'],
    'LWrist': ['LWrist'],
    'RShoulder': ['RShoulder', 'RElbow'],
    'RElbow': ['RElbow', 'RWrist'],
    'RWrist': ['RWrist']        
}
edge_parents = {
    'RHip': 'root', 'RKnee': 'RHip', 'RFoot': 'RKnee', 'LHip': 'root', 'LKnee': 'LHip', 'LFoot': 'LKnee',
    'Spine': 'root', 'Thorax': 'Spine','Neck': 'Thorax','Head': 'Neck',
    'LShoulder': 'Thorax','LElbow': 'LShoulder','LWrist': 'LElbow',
    'RShoulder': 'Thorax','RElbow': 'RShoulder', 'RWrist': 'RElbow'
}

joint_id_to_names = dict(zip(range(len(joint_names)), joint_names))
joint_name_to_id = {joint: i for i, joint in enumerate(joint_names)}
edge_id_to_names = dict(zip(range(len(edge_names)), edge_names))
edge_name_to_id = {edge: i for i, edge in enumerate(edge_names)}


def get_node_names(num_frames=3):
    node_names = []
    for i in range(num_frames):
        node_names += [name + str(i + 1) for name in joint_names]

    return node_names


def get_edge_names(num_frames=3):
    names = []
    for i in range(num_frames):
        names += [name + str(i + 1) for name in edge_names]

    return names
