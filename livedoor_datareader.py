import glob
from s2v import J2V
import numpy as np

class LivedoorReader:

    def __init__(self, filepath_to_livedoor='/Users/sochan/project/NLP/datas/livedoor'):
        self.filepath_to_livedoor = filepath_to_livedoor

    '''
    input: media_name (dokujo-tsushin, it-life-hack, kaden-channel ...)
    output: ['article', 'article2', ...] list of articles
    '''
    def __get_datas(self):
        media_datas = []
        for article in self.__get_article_names():
            with open('{}/{}/{}'.format(self.filepath_to_livedoor, self.media_name, article), 'r') as f:
                lines = f.readlines()[2:]
                media_datas.append(' '.join(lines))

        return media_datas

    def __get_article_names(self):
        return [ r.split('/')[-1] for r in glob.glob('{}/{}/*'.format(self.filepath_to_livedoor, self.media_name)) ]

    ''' returns encoded datas '''
    def __call__(self, media_name):
        j2v = J2V()
        self.media_name = media_name
        datas = self.__get_datas()
        encoded_datas = j2v.encode_sentences(datas)

        return encoded_datas


if __name__ == '__main__':
    lr = LivedoorReader('/Users/sochan/project/NLP/datas/livedoor')
    r = lr('smax')
    print(len(r[0]))
    print(len(r[0][0]))
