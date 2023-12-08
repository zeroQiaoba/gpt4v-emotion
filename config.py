
dataset2modality = {
    
    # facial emotion recognition
    'ckplus':    ['image'],
    'affectnet': ['image'],
    'fer2013':   ['image'],
    'rafdb':     ['image'],
    'sfew':      ['image'],

    # visual sentiment analysis
    'abstract':  ['evoke'],
    'artphoto':  ['evoke'],
    'twitter1':  ['evoke'],
    'twitter2':  ['evoke'],

    # micro-expression recognition
    'casme':     ['micro'],
    'casme2':    ['micro'],
    'samm':      ['micro'],

    # dynamic facial emotion recognition
    'dfew':      ['video'],
    'enterface': ['video'],
    'ferv39k':   ['video'],
    'ravdess':   ['video'],

    # multimodal emotion recognition
    'cmumosi':   ['text', 'video', 'multi'],
    'sims':      ['text', 'video', 'multi'],
    'mer2023':   ['text', 'video', 'multi'],

}

dataset2emos = {

    # facial emotion recognition
    'ckplus':    ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
    'affectnet': ['Surprise', 'Contempt', 'Happiness', 'Anger', 'Neutral', 'Sadness', 'Fear', 'Disgust'],
    'fer2013':   ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'],
    'rafdb':     ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'],
    'sfew':      ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],

    # visual sentiment analysis
    'abstract':  ['Amusement', 'Anger', 'Awe', 'Content', 'Disgust', 'Excitement', 'Fear', 'Sad'],
    'artphoto':  ['disgust', 'awe', 'sad', 'fear', 'anger', 'excitement', 'contentment', 'amusement'],
    'twitter1':  ['positive', 'negative'],
    'twitter2':  ['positive', 'negative'],

    # micro-expression recognition
    'casme':     ['tense', 'disgust', 'repression', 'surprise'],
    'casme2':    ['happiness', 'surprise', 'disgust', 'repression', 'others'],
    'samm':      ['Anger', 'Contempt', 'Happiness', 'Surprise', 'Other'],

    # dynamic facial emotion recognition
    'dfew':      ['Sad', 'Neutral', 'Angry', 'Fear', 'Surprise', 'Happy', 'Disgust'],
    'enterface': ['happiness', 'anger', 'disgust', 'fear', 'surprise', 'sadness'],
    'ferv39k':   ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    'ravdess':   ['surprised', 'neutral', 'disgust', 'sad', 'happy', 'calm', 'fearful', 'angry'],

    # multimodal emotion recognition
    'cmumosi':   ['negative', 'weakly negative', 'neutral', 'weakly positive', 'positive'],
    'sims':      ['negative', 'weakly negative', 'neutral', 'weakly positive', 'positive'],
    'mer2023':   ['worried', 'happy', 'neutral', 'angry', 'surprised', 'sad'],

}

# modality to [bsize and xishus]
modality2params = {
    'multi': [6,  [2, 3]],
    'video': [8,  [2, 2, 2]],
    'text':  [20, [2, 2, 5]],
    'image': [20, [2, 2, 5]],
    'evoke': [20, [2, 2, 5]],
    'micro': [20, [2, 2, 5]],
}

# change into your own keys
candidate_keys = []
