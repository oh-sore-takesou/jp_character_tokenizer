'''
文章を一つ受け取って名詞のみ'str str str'で返したい
'''
import MeCab
import mojimoji

class Parser:
    def __init__(self):
        self.tagger = MeCab.Tagger('-Ochasen')
        self.tagger.parse('')

    def __call__(self, s):
        node = self.tagger.parseToNode(s)
        list_owakati = []
        while node:
            word = node.surface
            features = node.feature.split(',')
            if features[0] == 'BOS/EOS':
                pass
            else:
                if features[7] != '*':
                    list_owakati.append(features[7])
                else:
                    list_owakati.append(' ')

            node = node.next

        return mojimoji.zen_to_han(''.join(list_owakati))


if __name__ == '__main__':
    parser = Parser()
    print(parser('私　はサッカーボールを蹴っています。'))
