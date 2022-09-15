
import logging
import time
import matplotlib
from telegram import Update, Bot, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters
import os
import csv
import matplotlib.pyplot as plt
matplotlib.use('Agg')
#matplotlib.use("TKAgg")

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

token = " " # Insert your token here

updater = Updater(token, use_context=True)
dp = updater.dispatcher

# -----------------------------------------------------------------------

# Local Variables

DATA_PATH = 'data/augmented/'
HISTORY_PATH = 'code/lazylossbot/history.txt'
PLOT_PATH = 'code/lazylossbot/plot.png'
GENERATED_PATH = 'data/generated/'

CHAT_ID = # Input your telegram chat ID

# -----------------------------------------------------------------------


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)



def automated_update(mode, epoch, loss, acc, time):

    time_unit = 's'

    if round(time, 0) > 100:
        # Conver to minutes
        time = round(time/60, 2)
        time_unit = 'min'
    elif round(time, 0) > 60:
        # Conver to hours
        time = round(time/60, 2)
        time_unit = 'h'

    if mode == 'default':
        if acc is not None:
            dp.bot.sendMessage(chat_id=CHAT_ID, text=f"- runtime ({time:.2f} {time_unit}) | epoch: [{epoch}] - loss: {loss:.2f} - acc: {acc:.2f}")
        else:
            dp.bot.sendMessage(chat_id=CHAT_ID, text=f"- runtime ({time:.2f} {time_unit}) | epoch: [{epoch}] - loss: {loss:.2f}")

    if mode == 'gan':
        dp.bot.sendMessage(chat_id=CHAT_ID, text=f"- runtime ({time:.2f} {time_unit}) | epoch: [{epoch}] - gen loss: {loss:.2f} - disc loss: {acc:.2f}")

    if mode == 'autoencoder':
        if acc is not None:
            dp.bot.sendMessage(chat_id=CHAT_ID, text=f"- runtime ({time:.2f} {time_unit}) | epoch: [{epoch}] - loss: {loss:.2f} - acc: {acc:.2f}")


def send_text(text):
    dp.bot.sendMessage(chat_id=CHAT_ID, text=text)


def send_update(update, context):

    history = from_txt()

    if len(history) == 0:
        update.message.reply_text(f"No history found")
        return

    epochs = []
    for i in history:
        epochs.append(i['epoch'])

    losses = []
    for i in history:
        losses.append(float(i['loss']))

    accuracies = []
    for i in history:
        accuracies.append(float(i['acc']))

    # Make a plot and save it
    img_file = PLOT_PATH
    if os.path.isfile(img_file):
        os.remove(img_file)

    plt.set_loglevel('WARNING')
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.plot(epochs, losses, label='gen loss', color='C0')
    ax2.plot(epochs, accuracies, label='disc loss', color='C1')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('gen loss')
    ax2.set_ylabel('disc loss')
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    fig.savefig(img_file)
    fig.clf()
    plt.clf()

    # update.message.reply_document(document=open(img_file, 'rb'))
    update.message.reply_photo(photo=open(img_file, 'rb'))

    update.message.reply_text(f'* current epoch: [{epochs[-1]}] - gen loss: {round(losses[-1],2)} - disc loss: {round(accuracies[-1],2)}')



def send_image(update, context):

    generated_images = GENERATED_PATH

    history = from_txt()

    if len(history) == 0:
        update.message.reply_text(f"No history found")
        return

    epochs = []
    for i in history:
        epochs.append(i['epoch'])

    newest_image = sorted(os.listdir(generated_images))[-1]

    # update.message.reply_document(document=open(generated_images + '/' + newest_image, 'rb'))
    update.message.reply_photo(photo=open(generated_images + newest_image, 'rb'))

    update.message.reply_text(f'* output at epoch [{epochs[-1]}]')


def test(update, context):

    update.message.reply_text(f"Working!")


# --------------------------------------------------------------------------------------------

# Extra function (not telegram related):

def from_txt():

    path = HISTORY_PATH

    history = []

    with open(path,'r') as file:
        reader = csv.DictReader(file)
        if reader != '':
            for row in reader:
                history.append(row)
    
    return history


def to_txt(epoch = 0, loss = 0, acc = 0):

    path = HISTORY_PATH

    # Check if the txt file has column headers
    if os.path.getsize(path) == 0:
        with open(path, 'w') as file:
            file.write('epoch,loss,acc\n')

    with open(path,'a', newline='') as file:
        writer = csv.writer(file, lineterminator='\r')
        writer.writerow([epoch, loss, acc])

def clear_txt():

    path = HISTORY_PATH

    with open(path, 'w') as file:
        file.write('epoch,loss,acc\n')

# --------------------------------------------------------------------------------------------

def main():

    dp.add_handler(CommandHandler("status", send_update))
    dp.add_handler(CommandHandler("send_image", send_image))
    dp.add_handler(CommandHandler("test", test))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
