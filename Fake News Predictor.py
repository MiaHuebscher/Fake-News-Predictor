'''
Project 1 
Mia Huebscher 
Fake News Predictor
'''

# Import necessary libraries to work with, analyze, and visualize the data 
import pandas as pd 
import time
import matplotlib.pyplot as plt
import sklearn.metrics as metrics 
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

# Set global variables for the csv files, the set of stopwords, and the string 
# Of various punctuations
FAKE_NEWS = 'Fake.csv'
TRUE_NEWS = 'True.csv'
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def plot_news_count(real_news_df, fake_news_df):
    '''
    Creates a bar chart that compares the number of real news articles in the 
    data against the number of fake news articles in the data

    Parameters
    ----------
    real_news_df : pandas dataframe 
        A dataframe filled with the data contained in 'True.csv'.
    fake_news_df : pandas dataframe
        A dataframe filled with the data contained in 'Fake.csv'.

    Returns
    -------
    None.

    '''
    # Create an empty dictionary that will map the possible categories of 
    # News articles (real/fake) to their respective amounts in the data
    plt_dict = {}
    
    # Calculate the amount of articles in each category and map them to their
    # Respective category
    real_news_count = len(real_news_df) 
    plt_dict['Real'] = real_news_count
    fake_news_count = len(fake_news_df)
    plt_dict['Fake'] = fake_news_count
    
    # Create a bar chart comparing the collected data of fake vs real news
    label = list(plt_dict.keys())
    counts = list(plt_dict.values())
    plt.bar(label, counts, color = ('gold', 'darkgreen'))
    plt.title('Number of Real News Articles vs Number of Fake News Articles')
    plt.ylabel('Number of Articles')
    plt.xlabel('Article Type')

def naive_bayes(news_df, given_article):
    '''
    Implements Multinomial Naive Baye's to construct a model that can make more
    accurate label predictions than the rest of my functions. 

    Parameters
    ----------
    news_df : dataframe 
        A dataframe that includes data for news articles both real and fake.
    given_article : string 
        A string of the article title given by the user.

    Returns
    -------
    predicted_label : string
        A string that denotes the label of the given article that was predicted
        using the Naive Baye's model.
    '''
    # Create a smaller data frame that just includes each news article's
    # Title and label
    news_txt_df = news_df[['title', 'label']]
    
    # Split the data frame into data for training and data for testing
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(news_txt_df['title']) 
    x_train, x_test, y_train, y_test = train_test_split(X, news_txt_df['label'],
                                                        test_size=0.2, 
                                                        random_state = 1)
    
    # Train the Naive Baye's model using the training data and then make label 
    # Predictions using the testing data 
    nb = MultinomialNB()
    model = nb.fit(x_train, y_train)
    prediction = model.predict(x_test)
    
    # Calculate the accuracy of this model 
    score = metrics.accuracy_score(y_test, prediction)
    print("Accuracy: ", round((score*100), 3), '%')

    # Create a confusion matrix to illustrate the specific accuracies and 
    # Failures of the model 
    cm = confusion_matrix(y_test, prediction)
    
    cm_df = pd.DataFrame(cm, columns = np.unique(y_test), 
                         index = np.unique(y_test))     
    cm_df.columns.name = 'Predicted Label'
    cm_df.index.name = 'Actual Label'
    
    sns.heatmap(cm_df, annot=True, square=True, cmap="Reds",annot_kws={"size":12}, 
                fmt='g')
    
    # Create a dataframe that only includes the given article title, then 
    # Vectorize this title so the naive baye's model can work with it
    article_df = pd.DataFrame()
    article_df['title'] = [given_article]
    article = vectorizer.transform(article_df['title'])
    
    # Have the model predict whether the given article is real or fake, this 
    # Will, on average, be the most most accurate prediction of all my functions
    predicted_label = model.predict(article)
    
    return predicted_label
   
def avrg_title_length(df):
    '''
    Calculates the average length of all the titles in the 'title' column of the 
    given dataframe 

    Parameters
    ----------
    df : dataframe
        A dataframe that includes a column for the titles of articles.

    Returns
    -------
    title_avrg : float
        The average length of a title in the dataframe's 'title' column.
    title_stdev : float
        The standard deviation for the 'title' column data.
    title_lengths : list
        A list of the lengths of each title within the dataframe.

    '''
    # Create an empty list
    title_lengths = []
    
    # Add the length of each title in the dataframe to the above list
    for index, row in df.iterrows():
        title_words = row['title'].split()
        title_lengths.append(len(title_words))
    
    # Calculate the average of all the values in title_lengths 
    title_avrg = sum(title_lengths) / len(title_lengths)
    
    # Compute the standard deviation of the title length data
    title_stdev = np.std(title_lengths)
    
    return title_avrg, title_stdev, title_lengths 

def plot_title_lengths(real_title_avrg, real_title_std, real_title_lngth, 
                     fake_title_avrg, fake_title_std, fake_title_lngth):
    '''
    Creates a bar chart that compares the average length of a real news title 
    vs the average length of a fake news title

    Parameters
    ----------
    real_title_avrg : float
        The average length of a real news title.
    real_title_std : float 
        The standard deviation for real news title lengths.
    real_title_lngth : list
        A list of the lengths of each real news title in the data.
    fake_title_avrg : float
        The average length of a fake news title.
    fake_title_std : float
        The standard deviation for fake news title lengths.
    fake_title_lngth : list 
        A list of the lengths of each fake news title in the data.

    Returns
    -------
    None.

    '''
    
    # Compute the standard errors to plot in the bar chart 
    real_error = real_title_std / np.sqrt(len(real_title_lngth))
    fake_error = fake_title_std / np.sqrt(len(fake_title_lngth))
    
    # Create the necessary lists of data to use in the bar chart 
    means = [real_title_avrg, fake_title_avrg]
    error = [real_error, fake_error]

    # Create the bar chart
    plt.bar(['Real', 'Fake'], means, color = ('gold', 'darkgreen'), 
            yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.title("Average Number of Words in Real News Titles vs in Fake News Titles")
    plt.ylabel('Average Number of Words in Title')
    plt.xlabel('Type of News')
            
def get_word_count(df):
    '''
    Creates a dictionary of all the words present in the given dataframe's 
    'title' column. Map these words to their number of occurrences among all 
    the titles in the dataframe

    Parameters
    ----------
    df : dataframe
        A dataframe with a 'title' column.

    Returns
    -------
    word_counts :dictionary
        A dictionary that has words as keys and their total occurrences throughout 
        the dataframe's 'title' column as values.
    '''
    # Create an empty dictionary 
    word_counts = {}
    # Iterate through the dataframe and look at the titles for each row
    for i in range(len(df)):
        title = df.iloc[i, 0]
        # Get rid of the punctuations so that there are no unncessary repeats
        # In the dictionary (i.e. {Trump:2, Trump's:1})
        new_title = ''
        for char in title:
            if char not in PUNCTUATIONS:
                new_title += char
        title_lst = new_title.split(' ')
        # Each time a word occurs, either create a new key for it or increase 
        # The word count for the word
        for word in title_lst:
            if len(word) > 1: 
                if word[1] == word[1].upper():
                    new_word = word 
                else:
                    new_word = word.lower()
            else:
                new_word = word.lower()
                
            if new_word not in word_counts:
                word_count = 0 
                word_count += 1 
                word_counts[new_word] = word_count
            else: 
                word_counts[new_word] += 1 
    
    return word_counts

def sig_words(frst_dct, sec_dct):
    '''
    Filters a dictionary of words to only include significant words. This will 
    allow us to see which meaningful words are seen more in fake news titles 
    vs real news titles

    Parameters
    ----------
    frst_dct : dictionary 
        A dictionary returned by the get_word_count function.
    sec_dct : dictionary 
        A dictionary returned by the get_word_count function.

    Returns
    -------
    frst_dct : dictionary
        The same input dictionary, except without words deemed insignificant.
    sec_dct : dictionary
        The same input dictionary, except without words deemed insignificant.
    '''
    # Create a list that will contain words to delete from each dictionary 
    del_lst = []
    # Add an empty string to the del_lst so that empty strings do not show up 
    # In the dictionaries 
    del_lst.append('')
    
    for word in frst_dct.keys():
        # Add words in all caps to the del_list; those will be worked with 
        # In a seperate function
        if word == word.upper():
            del_lst.append(word)
        # Add stopwords to del_lst because they are too common
        if word.lower() in STOPWORDS:
            del_lst.append(word)
        
        # Add unique words in one dictionary to the other dictionary, this will
        # Allow us to see if real news titles use specific words that fake 
        # News titles don't and vice versa
        if word not in sec_dct.keys():
            sec_dct[word] = 0
    
    # Repeat the processes in the above for-loop, but this time using sec_dct
    # As a reference for words
    for word in sec_dct.keys():
        if word == word.upper():
            del_lst.append(word)
        if word.lower() in STOPWORDS:
            del_lst.append(word)
        if word not in frst_dct.keys():
            frst_dct[word] = 0
     
    # Delete each word in del_lst from both dictionaries to ensure each
    # Dictionary has the same keys
    for word in del_lst:
        if word in frst_dct.keys():
            del frst_dct[word]
        if word in sec_dct.keys():
            del sec_dct[word]     
                
    return frst_dct, sec_dct

def get_freq_dct(real_dct, fake_dct):
    '''
    Mutate each dictionary returned above so that they include frequency 
    amounts as values instead of word counts. Frequency is calculated as the 
    number of times a word appears in one dictionary divided by the total 
    number of times it appears in both dictionaries

    Parameters
    ----------
    real_dct : dictionary 
        A dictionary of words mapped to their amount of appearances in real 
        news titles.
    fake_dct : dictionary
        A dictionary of words mapped to their amount of appearances in fake 
        news titles.

    Returns
    -------
    sorted_rl_dct : dictionary
        A dictionary with words as keys and values as their frequencies. The 
        dictionary is sorted in descending order.
    sorted_fk_dct : dictionary
        A dictionary with words as keys and values as their frequencies. The 
        dictionary is sorted in descending order
    '''
    # Create an emtpy dictionary 
    real_freq_dct = {}
    
    for word in real_dct.keys():
        # Do not include words that have a low total occurrence, as they are not
        # Entirely relevant
        if real_dct[word] + fake_dct[word] > 30:
            # Compute the frequencies for each word and add them as values to 
            # The new dictionary 
            rl_wd_freq = round(real_dct[word] / (real_dct[word] + fake_dct[word]), 2)
            real_freq_dct[word] = rl_wd_freq
    
    # Follow the same steps as above, but this time with data from the
    # Dictionary with data for fake news words 
    fake_freq_dct = {}
    for word in fake_dct.keys():
        if real_dct[word] + fake_dct[word] > 30:
            fk_wd_freq = round(fake_dct[word] / (real_dct[word] + fake_dct[word]), 2)
            fake_freq_dct[word] = fk_wd_freq
    
    # Sort the dictionaries in descending order
    sorted_rl_dct = {k:v for k, v in sorted(real_freq_dct.items(), 
                                        key = lambda item:item[1], 
                                        reverse = True)}
    
    sorted_fk_dct = {k:v for k, v in sorted(fake_freq_dct.items(), 
                                        key = lambda item:item[1], 
                                        reverse = True)}
    
    return sorted_rl_dct, sorted_fk_dct

def filter_freq_dct(dct):
    '''
    Filters the frequency dictionary so that it only includes words with 
    relevant frequencies

    Parameters
    ----------
    dct : dictionary 
        A dictionary with words mapped to their frequencies.

    Returns
    -------
    new_dct : dictionary
        A dictionary only including words with relevant frequencies.
    '''
    # Create an empty dictionary 
    new_dct = {}
    
    # Iterate through the given dictionary and only add words to the new 
    # Dictionary that have frequencies over 70%, these are the most significant
    for key in dct.keys():
        if dct[key] >= 0.70:
            new_dct[key] = dct[key]
  
    return new_dct

def plt_wd_freq(freq_dct, color, wds_lst):
    '''
    Creates a bar plot that shows words and their respective frequencies.
    These words were manually picked so that the bar chart only includes words
    that have high frequencies. 

    Parameters
    ----------
    freq_dct : dictionary
        A dictionary with words as keys and their frequencies as values.
    color : string
        The color the bars in the bar chart will be plotted in.
    wds_lst : list
        A list of words to include in the bar chart.

    Returns
    -------
    None.

    '''
    # Create an empty list
    freqs = []
    
    # For every word in word_lst, add their frequencies to the list, freqs
    for word in wds_lst:
        freq = freq_dct[word]
        freqs.append(freq)
    
    # Create a bar chart that illustrates the frequencies of each word
    plt.bar(wds_lst, freqs, color = color)
    plt.xlabel('Words')
    plt.xticks(rotation = 70)
          
def capitalization_count(df):
    '''
    Returns a list of the number of words in each title in a dataframe that 
    are written in all capital letters 

    Parameters
    ----------
    df : dataframe
        A dataframe with a 'title' column.

    Returns
    -------
    num_caps_lst : list 
        A list that contains the number of words in each title in a dataframe 
        that are written in all capital letters. 
    '''
    # Create an empty list to append to    
    num_caps_lst = [] 
    
    # Iterate through the dataframe and pull out the title words for each row
    for i in range(len(df)):
        title = df.iloc[i, 0]
        title_lst = title.split(' ')
        # Count the number of words in all capital letters in a single title
        num_caps = 0 
        for word in title_lst:
            if word == word.upper():
                num_caps += 1 
        # Append this number to the list, num_caps_lst
        num_caps_lst.append(num_caps)

    return num_caps_lst 

def plot_cap_num(real_caps_lst, fake_caps_lst):
    '''
    Creates a bar plot comparing the average length of a title in fake news 
    articles vs real news articles

    Parameters
    ----------
    real_caps_lst : list 
        The list returned by the function capitalization_count using data from
        real news articles.
    fake_caps_lst : list
        The list returned by the function capitalization_count using data from
        fake news articles.

    Returns
    -------
    None.
    '''
    
    # Compute the average length of a title for real news and fake news
    real_avrg_cap = sum(real_caps_lst) / len(real_caps_lst)
    fake_avrg_cap = sum(fake_caps_lst) / len(fake_caps_lst)
    
    # Calculate the standard deviation for title lengths of real and fake news
    real_cap_std = np.std(real_caps_lst)
    fake_cap_std = np.std(fake_caps_lst)

    # Compute the standard error for real and fake news titles
    real_error = real_cap_std / np.sqrt(len(real_caps_lst))
    fake_error = fake_cap_std / np.sqrt(len(fake_caps_lst))
    
    # Compile the computed averages and errors into lists to use in the bar plot
    means = [real_avrg_cap, fake_avrg_cap]
    error = [real_error, fake_error]

    # Create the bar plot
    plt.bar(['Real', 'Fake'], means, color = ('gold', 'darkgreen'), 
            yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.title("Average Number of Words in Capital Letters in Real New's Titles"
              " vs in Fake New's Titles")
    plt.ylabel('Average Number of Capital Words in Title')
    plt.xlabel('Type of News')
    
def get_pos_tags(df):
    '''
    Creates a dictionary of the average number of specific parts of speech 
    (Proper Nouns, Plural Proper Nouns, Nouns, Plural Nouns, and Past-Tense
    Verbs) in a fake news title or real news title (depends on the given 
    dataframe). Function only runs accurately if given titles that do not
    automatically capitalize the first letter of every word. Created to verify 
    an article that stated that fake news titles tend to have more proper nouns, 
    more past tense verbs, and less nouns over all. 
    
    Parameters
    ----------
    df : dataframe
        A dataframe with a title column.

    Returns
    -------
    avrg_sig_pos_tags : dictionary 
        A dictionary with keys as parts of speech and values as their average 
        occurrence within a title.
    errors_dct : dictionary
        A dictionary with keys as parts of speech and values as the standard 
        error of their occcurrence within a title
    '''
    # Create a dictionary that has keys of parts of speech we care about and 
    # Values as empty lists
    full_sig_pos_tags = {'NNP':[], 'NNPS':[], 'NN':[], 'NNS':[], 'VBD':[]}
    
    for i in range(len(df)):
        # For each row of the dataframe, create an empty dictionary to keep
        # Track of the number of occurrences of each part of speech in each 
        # Title of the dataframe
        sig_pos_tags = {}
        
        # Initialize the parts of speech counts to zero
        sig_pos_tags['NNP'] = 0 
        sig_pos_tags['NNPS'] = 0 
        sig_pos_tags['NN'] = 0 
        sig_pos_tags['NNS'] = 0 
        sig_pos_tags['VBD'] = 0 
        
        # Extract the title from the current row in the dataframe and split it
        # Into a list of its individual sentences
        title = df.iloc[i, 0]
        title_sents = nltk.sent_tokenize(title)
        
        # For every sentence in the current title, split the sentence into 
        # A list of its words
        for sent in title_sents:
            words = nltk.word_tokenize(sent)
            
            # Filter the list of words so it does not include stopwords
            words = [word for word in words if word not in STOPWORDS]
            
            # Get the tag for each word
            tagged_wds = nltk.pos_tag(words)
            
            # Iterate through the tags of each word and count their number of 
            # Occurrences
            for word, tag in tagged_wds:
                if tag == 'NNP':
                    sig_pos_tags['NNP'] += 1 
                elif tag == 'NNPS':
                    sig_pos_tags['NNPS'] += 1  
                elif tag == 'NN':
                    sig_pos_tags['NN'] += 1  
                elif tag == 'NNS':
                    sig_pos_tags['NNS'] += 1  
                elif tag == 'VBD':
                    sig_pos_tags['VBD'] += 1 
        
        # Add each part of speech's total number of occurrence throughout the 
        # Title to their respective lists in full_sig_pos_tags
        full_sig_pos_tags['NNP'].append(sig_pos_tags['NNP'])
        full_sig_pos_tags['NNPS'].append(sig_pos_tags['NNPS'])
        full_sig_pos_tags['NN'].append(sig_pos_tags['NN'])
        full_sig_pos_tags['NNS'].append(sig_pos_tags['NNS'])
        full_sig_pos_tags['VBD'].append(sig_pos_tags['VBD'])
    
    # Create two empty dictionaries that track the errors and averages of 
    # Each part of speech's number of occurrences in a title
    errors_dct = {}
    avrg_sig_pos_tags = {}
    
    # For each part of speech, calculate its average occurrence in a title and
    # The error of its average. Add the calculations to the respective 
    # Dictionaries
    for key, value in full_sig_pos_tags.items():
        avrg = sum(value) / len(value)
        avrg_sig_pos_tags[key] = avrg
        stdev = np.std(value)
        error = stdev / np.sqrt(len(value))
        errors_dct[key] = error
    
    return avrg_sig_pos_tags, errors_dct

def plt_pos_tags(rl_avrg_pos_tags, fk_avrg_pos_tags, rl_errors_dct, 
                 fk_errors_dct):
    '''
    Creates a bar plot comparing the average number of occurrences for each 
    part of speech in a fake news title versus in a real news title

    Parameters
    ----------
    rl_avrg_pos_tags : dictionary
        The dictionary of averages returned by get_pos_tags when given a
        dataframe of real news data.
    fk_avrg_pos_tags : dictionary
        The dictionary of averages returned by get_pos_tags when given a
        dataframe of fake news data.
    rl_errors_dct : dictionary
        The dictionary of errors returned by get_pos_tags when given a
        dataframe of real news data.
    fk_errors_dct : dictionary
        The dictionary of errors returned by get_pos_tags when given a
        dataframe of fake news data.

    Returns
    -------
    None.
    '''
    # Make a list containing the parts of speech that will occur on the graph
    x = ['Proper Noun', 'Proper Noun, plural', 'Noun', 'Noun, plural', 
         'Verb, past tense']
    
    # Create empty lists that will contain the average number of occurrence 
    # For each part of speech in fake news titles and real news titles 
    real_y = []
    fake_y = []
    
    # Create more lists that will contain the errors of each average for real 
    # News part of speech data and fake news part of speech data 
    rl_errors = []
    fk_errors = []
    
    # Iterate through the parameter dictionaries given to the function and
    # Append the correct values to their corresponding lists above
    for key, value in rl_avrg_pos_tags.items():
        real_y.append(value)
        
    for key, value in fk_avrg_pos_tags.items():
        fake_y.append(value)
        
    for key, value in rl_errors_dct.items():
        rl_errors.append(value)
    
    for key, value in fk_errors_dct.items():
        fk_errors.append(value)
    
    # Arrange the x-axis values
    x_axis = np.arange(len(x))
    
    # Create the bar graph
    plt.bar(x_axis - 0.2, real_y, 0.4, color = 'gold', yerr = rl_errors, 
            label = 'real news')
    plt.bar(x_axis + 0.2, fake_y, 0.4, color = 'darkgreen', yerr = fk_errors,
            label = 'fake news')
    plt.legend()
    plt.xticks(x_axis, x, rotation = 45)    
    plt.xlabel('Parts of Speech')
    plt.ylabel('Average Number in News Title')
    plt.title('The Average Number of Specific Parts of Speech in Fake News'
              ' Titles vs Real News Titles')
          
if __name__ == '__main__':
    
    # Create two data frames, one for real news articles and another for fake 
    # News articles 
    real_news_df = pd.read_csv(TRUE_NEWS, usecols = ['title','text', 'subject', 
                                                     'date'])
    fake_news_df = pd.read_csv(FAKE_NEWS, usecols = ['title','text', 'subject', 
                                                     'date'])
    
    # Drop the rows containing null values from both data frames
    real_news_df = real_news_df.dropna()
    fake_news_df = fake_news_df.dropna()
    
    # Combine the two data sets into one data frame, with an added column that 
    # Distinguishes each news article as true or fake 
    real_news_df['label'] = 'Real'
    fake_news_df['label'] = 'Fake'
    news_df = pd.concat([real_news_df, fake_news_df])
    
    # Quick Analysis of Data
    print('Here is a quick analysis of the data used to train this code:')
    
    # Plot a bar graph comparing fake news counts to real news counts to ensure
    # Our data is well-balanced and thus is able to provide for better analysis
    plt.figure(1)
    plot_news_count(real_news_df, fake_news_df)
    plt.show()
    time.sleep(4)
    
    # Create a bar graph comparing the average number of words in each news type's
    # Title. Because there is a statistical difference in title lengths, the 
    # Length of an article's title can help us predict if the article is real 
    # Or fake
    real_ttl_avrg, real_ttl_std, real_ttl_lngth = avrg_title_length(real_news_df)
    fake_ttl_avrg, fake_ttl_std, fake_ttl_lngth = avrg_title_length(fake_news_df)
    
    plt.figure(2)
    plot_title_lengths(real_ttl_avrg, real_ttl_std, real_ttl_lngth, 
                     fake_ttl_avrg, fake_ttl_std, fake_ttl_lngth)
    plt.show()
    time.sleep(4)
    
    # Get word frequency for each significant word in each news type's titles
    real_word_dct = get_word_count(real_news_df)
    fake_word_dct = get_word_count(fake_news_df)
    real_dct, fake_dct = sig_words(real_word_dct, fake_word_dct)
    real_freq_dct, fake_freq_dct = get_freq_dct(real_dct, fake_dct)   
    
    # Create two bar charts. One contains the frequency of specific words 
    # Occuring in real news titles. The other contains the frequency of 
    # Different specific words occuring in fake news titles. These specific
    # Words were manually picked by me to show the key differences in vocabulary 
    # And grammar among real news titles and fake news titles. 
    plt.figure(3, figsize = (10,4))
    
    # Manually give the plt_wd_freq function a list of words to plot. This is 
    # Because some words in our frequency dicts are common words like 'a' or  
    # 'trump' that are not efficient/accurate in determining the realness or 
    # Fakeness of a news article 
    real_bar_lst = ['opposition', 'reforms', 'urges', 'looms', 'coalition', 
                    'arms', 'policies', 'unlikely', 'gas', 'resolution', 
                    'europe','syria', 'financial', 'london', 'advisers', 
                    'poland', 'aides', 'accusations', 'tech', 'russia', 
                    'healthcare', 'coup', 'congress', 'testify', 'asylum']
    
    plt_wd_freq(real_freq_dct, 'gold', real_bar_lst)
    plt.ylabel('Frequency of Occurrence in Real News Titles')
    plt.title('Frequency of Occurrence for Specific Words in Real News Titles')
    plt.show()
    time.sleep(4)

    plt.figure(4, figsize = (10,4))
    fake_bar_lst = ['cops', 'dems', 'went', 'rant', 'video', 'insane', 'lies', 
                    'liberal', 'hollywood', 'truth', 'gun', 'biggest', 'scam', 
                    'knew', 'woman', 'asked', 'american', 'school', 'gay',
                    'nasty', 'hitler', 'history', 'propaganda', 'hispanic', 'told']
    
    plt_wd_freq(fake_freq_dct, 'darkgreen', fake_bar_lst)
    plt.ylabel('Frequency of Occurrence in Fake News Titles')
    plt.title('Frequency of Occurrence for Specific Words in Fake News Titles')
    plt.show()  
    time.sleep(4)
    
    print()
    print('After analyzing the differences in the specific words that occur'
          ' most in real news titles vs in fake news titles, several conjectures'
          ' arise. One reasonable claim to be made is that fake news titles'
          ' tend to contain more words pertaining to American society and history,'
          ' while real news titles tend to possess more words related to current'
          ' political affairs. Furthermore, words within real news titles'
          ' tend to be longer and more complex. Real news titles also seem to'
          ' report more on economic and political conditions, while fake news'
          ' titles seem to reflect on opinions regarding society. Fake news'
          ' titles also contain more past tense verbs. These key differences'
          ' may enable us to predict an article real or fake based off of the'
          ' contents of its title')
    time.sleep(4)
    print()
    
    # Filter the word frequency dictionaries so that they only include words 
    # With relevant frequencies that help us decide if a news article is real
    # Or fake based off the specific words used in its title
    real_plt_dct = filter_freq_dct(real_freq_dct)
    fake_plt_dct = filter_freq_dct(fake_freq_dct)
    
    # Plot a bar chart reporting the average number of full capital-letter words
    # In fake news titles vs real news titles. Because there is a statistical 
    # Difference, the number of full-capital-letter words can be another factor 
    # In determining whether news is real or fake 
    real_caps_lst = capitalization_count(real_news_df)
    fake_caps_lst = capitalization_count(fake_news_df)
    
    plt.figure(5)
    plot_cap_num(real_caps_lst, fake_caps_lst)
    plt.show()
    
    # Create a bar plot that illustrates the average frequencies of occurrence  
    # For specific parts of speech in a real news title versus fake news title.
    # This bar graph is only accurate when get_pos_tags is provided a dataframe
    # That has 'title' column values that do not begin every word with a
    # Capital letter 
    rl_avrg_pos_tags, rl_errors_dct = get_pos_tags(real_news_df)
    fk_avrg_pos_tags, fk_errors_dct = get_pos_tags(fake_news_df)
    
    plt.figure(6)
    plt_pos_tags(rl_avrg_pos_tags, fk_avrg_pos_tags, rl_errors_dct, 
                 fk_errors_dct)
    plt.show()  
    
    # Ask the user for the title of article they want to see is true or fake
    given_article = str(input('Please enter the title of a news article that you'
                         ' wish to determine True or Fake: '))
    print()
    
    
    # Create a list that will contain the individual predictions returned using
    # Three seperate techniques 
    label_pred_lst = []
    
    # Create a dataframe that includes the given title
    data = {'Title': [given_article]}
    given_article_df = pd.DataFrame(data)
    
    # Split the given title into a list of its words
    given_title_wds = given_article.split()
    
    # Make a prediction based on title length and append it to label_pred_lst
    print('Prediction by Title Length:')
    given_title_lngth = len(given_title_wds)
    
    # Calculate the difference between the given title length and the upper 
    # Error bar for real news and the difference between the given title length
    # And the lower error bar for fake news
    real_ttl_error = real_ttl_std / np.sqrt(len(real_ttl_lngth))
    fake_ttl_error = fake_ttl_std / np.sqrt(len(fake_ttl_lngth))
    real_lngth_max = real_ttl_avrg + real_ttl_error
    fake_lngth_min = fake_ttl_avrg - fake_ttl_error
    
    diff_from_real = abs(given_title_lngth - real_lngth_max)
    diff_from_fake = abs(given_title_lngth - fake_lngth_min)
    
    # Whichever error bar (upper for real news and lower for fake news) is 
    # Closer to the title length determines the prediction label
    if diff_from_fake > diff_from_real:
        label_pred_lst.append('True')
        print("Based on the length of your article's title, your article has"
              " been predicted true")
    elif diff_from_fake < diff_from_real:
        label_pred_lst.append('Fake')
        print("Based on the length of your article's title, your article has"
              " been predicted fake")
    # If the distances are equal, inform the user that a prediction could not 
    # Be made
    else:
        print("Sorry no prediction can be made using the length of your"
              " article's title")
        
       
    # Make a prediction based on title content and append it to label_pred_lst
    print()
    print('Prediction by Title Content:')
    
    # Create two empty lists
    real_lst = []
    fake_lst = []
    
    # Iterate through every word in the given tile
    for word in given_title_wds:
        # If the word is in the dicrtionary of frequent real news words, append
        # The word to real_lst
        if word.lower() in real_plt_dct:
            real_lst.append(word)
        # Follow the same steps if the word is in the dict of fake news words
        elif word.lower() in fake_plt_dct:
            fake_lst.append(word)
            
    # Make the prediction based off which list is longer 
    if len(real_lst) > len(fake_lst):
        label_pred_lst.append('True')
        print("Based on the content of your article's title, your article has"
              " been predicted true")
    elif len(real_lst) < len(fake_lst):
        label_pred_lst.append('Fake')
        print("Based on the content of your article's title, your article has"
              " been predicted fake")
    else:
        print("Sorry no prediction can be made using the content of your"
              " article's title")       
    
    # Prediction by number of words with only capital letters and append it to 
    # Label_pred_lst
    print()
    print('Prediction by the Number of Words in All Capital Letters:')
    
    # Calculate the number of all capital words in the given title 
    given_num_capitals = capitalization_count(given_article_df)
    given_num_capitals = given_num_capitals[0]
    
    # Upon analyzing the titles of real news and fake news, I felt that the
    # Most accurate way to make this prediction is by categorizing new titles 
    # With one or less all capital letter words as real news. Clearly, this 
    # Approach does not yield the best accurracy
    if given_num_capitals <= 1:
        label_pred_lst.append('True')
        print("Based on the number of full capital letter words in your"
              " article's title, your article has been predicted true")
    else:
        label_pred_lst.append('Fake')
        print("Based on the number of full capital letter words in your"
              " article's title, your article has been predicted fake")
    
    print()
    # Initialize variables
    real_count = 0 
    fake_count = 0 
    
    # Calculate the number of predictions that returned true and the number 
    # Of predictions that returned false
    for label in label_pred_lst:
        if label == "True":
            real_count += 1 
        else:
            fake_count += 1
    
    # Whichever prediction was returned more, is the final prediction made by 
    # The code
    if real_count > fake_count:
        print("After running several analyses on your article's title, the"
              " majority of the models within this code have predicted your"
              " article as True")
    elif fake_count > real_count:
        print("After running several analyses on your article's title, the"
              " majority of the models within this code have predicted your"
              " article as Fake")
    else: 
        print("After running several analyses on your article's title, it"
              " remains uncertain whether your article is true or fake")

   
    # Make the prediction based off my MultiNomial Naive Baye's model. 
    print()
    print("The following confusion matrix depicts the accuracy of my Naive"
          " Baye's model in detail:")
    plt.figure(7)
    prediction = naive_bayes(news_df, given_article)
    plt.show()
    print()
    if prediction[0] == 'Fake':
        print("With a 94.911% accuracy, the Naive Baye's model predicts your"
              " article as being Fake")
    elif prediction[0] == 'Real':
        print("With a 94.911% accuracy, the Naive Baye's model predicts your"
              " article as being Fake")