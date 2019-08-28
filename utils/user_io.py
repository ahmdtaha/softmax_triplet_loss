

def ask_yes_no_question(question):
    print(question+ ' [y/n] ')
    while True:
        answer = input()
        if answer.lower() in ['y','yes']:
            return True
        elif answer.lower() in ['n','no']:
            return False
        print('Please Enter a valid answer')


if __name__ == '__main__':
    ask_yes_no_question('File Exists, Continue ? ')