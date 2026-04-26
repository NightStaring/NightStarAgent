import yaml
import os


class UserQueryPrompt:

    def __init__(self):
        self.prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.yaml')
        self.prompt = yaml.load(open(self.prompt_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    # 默认
    def get_default_prompt(self, question: str):
        return self.prompt['user_query']['query']['default']['llm'].format(question=question)

    # 选择
    def get_rag_query_check_prompt(self, question: str):
        return self.prompt['user_query']['query']['check']['rag'].format(question=question)

    def get_image_rag_query_check_prompt(self, question: str, image_desc: str):
        return self.prompt['user_query']['query']['check']['image_rag'].format(question=question, image_desc=image_desc)

    # 扩展
    def get_extend_question_prompt(self, question: str):
        return self.prompt['user_query']['query']['extend']['question'].format(question=question)

    def get_extend_words_prompt(self, question: str):
        return self.prompt['user_query']['query']['extend']['words'].format(question=question)

    # 思考
    def get_pre_think_prompt(self, question: str):
        return self.prompt['user_query']['think']['pre'].format(question=question)


class AnswerPrompt:
    def __init__(self):
        self.prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.yaml')
        self.prompt = yaml.load(open(self.prompt_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    def get_chunk_answer_prompt(self, context: str, question: str):
        return self.prompt['answer']['chunk'].format(context=context, question=question)

    def get_over_check_prompt(self, question: str, sub_questions: str, context: str):
        return self.prompt['answer']['over_check'].format(question=question, sub_questions=sub_questions, context=context)

    def get_text_answer_prompt(self, context: str, question: str):
        return self.prompt['answer']['text'].format(context=context, question=question)

    def get_image_answer_prompt(self, context: str, question: str):
        return self.prompt['answer']['image'].format(context=context, question=question)

def get_user_query_prompt():
    return UserQueryPrompt()

def get_answer_prompt():
    return AnswerPrompt()