from HMM import HMM
import alarm
import carnet

if __name__ == '__main__':
    model = HMM()
    model.load('partofspeech.browntags.trained')

    # Problem 2

    print("\n----------Generate----------")
    print(model.generate(10))

    print("\n----------Forward----------")
    model.parse_file_to_observations('ambiguous_sents.obs')

    print("\n----------Viterbi----------")
    model.parse_file_to_observations('ambiguous_sents.obs', 'viterbi')

    # Problem 3
    print("\n----------Alarm----------")
    alarm.run_alarm()
    print("\n----------Car----------")
    carnet.run_car()
