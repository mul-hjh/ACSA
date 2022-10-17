import matplotlib.pyplot as plt
import os
import cv2

# visualize loss & accuracy
def visualize_curve(log_root):
    log_file = open(log_root, 'r')
    # print(log_file.readline())
    # print(log_file.readlines())
    result_root = log_root[:log_root.rfind('/') + 1] + 'test.jpg'
    loss = []
    epochs = []
    
    top1_i2t = []
    top10_i2t = []
    top1_t2i = []
    top10_t2i = []
    for line in log_file.readlines():
        # print("hi")
        # print(line)
        line = line.strip().split()
        # print(line)
        if len(line) == 4:
            loss.append(line[3])
        if len(line) == 8:
            top1_t2i.append(line[1])
        #
        #
        #

        # loss.append(line[5])
        # top1_i2t.append(line[1])
        # top10_i2t.append(line[3])
        # top1_t2i.append(line[7])
        # top10_t2i.append(line[9])
    # print(len(loss))
    log_file.close()

    x1 = range(0,20)
    # x2 = range(1,100)
    plt.title('loss')
    # plt.figure('loss')
    plt.plot( x1, loss, '.-')
    plt.yticks([0,0.2,0.5,1,2,5,10,20,30,40,50])

    # plt.figure('accuracy')
    # plt.subplot(211)
    # plt.plot(top1_i2t, label = 'top1')
    # plt.plot(top10_i2t, label = 'top10')
    # plt.legend(['image to text'], loc = 'upper right')
    # plt.subplot(212)
    # plt.plot(top1_t2i, label = 'top1')
    # plt.plot(top10_i2t, label = 'top10')
    # plt.legend(['text to image'], loc = 'upper right')
    # plt.savefig(result_root)
    plt.show()


if __name__ == '__main__':
    log_root = '/workspace2/junhua/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching-master/Results/multi-task/ssl_0.2/logs/lr-0.0002-decay-0.9-batch-16/train.log'
    visualize_curve(log_root)
