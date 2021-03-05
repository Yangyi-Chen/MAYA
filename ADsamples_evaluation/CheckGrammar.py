import language_tool_python


class GrammarChecker():
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')



    def check(self, sentence):
        '''
        :param sentence:  a string
        :return:
        '''
        matches = self.lang_tool.check(sentence)
        return len(matches)


if __name__ == '__main__':
    checker = GrammarChecker()
    print(checker.check('he plant a tree'))
