import pandas as pd
import matplotlib.pyplot as plt


def question_1():
    df1 = pd.read_csv('Olympics_dataset1.csv', skiprows=1, thousands=',')
    df2 = pd.read_csv('Olympics_dataset2.csv', skiprows=1, thousands=',')
    df = pd.merge(df1, df2, how='left', on='Unnamed: 0').rename(columns={'Unnamed: 0': 'Country'})
    print("Question 1: \n", df.head(5))
    return df


def question_2(df):
    q2 = df.set_index('Country')
    print("Question 2: \n", q2.head(1))


def question_3(df):
    q3 = df.drop(columns=['Rubish'])
    print("Question 3: \n", q3.head(5))


def question_4(df):
    q4 = df.dropna()
    print("Question 4: \n", q4.tail(10))


def question_5(df):
    temp = df.dropna(subset=['Gold_x'])
    temp = temp.drop(temp.index[-1])
    q5 = temp.sort_values(by='Gold_x', ascending=False)
    print("Question 5: \n", q5.head(1))
    '''
    # another method
    temp = list(df['Gold.1'])
    temp.pop()
    for i in range(len(temp)):
        if not isinstance(temp[i],float):
            temp[i] = re.sub(r',', '', temp[i])
    temp = [float(x) for x in temp]
    index = temp.index(max(temp))
    print(df['Unnamed: 0'][index])
    '''


def question_6(df):
    temp = df.dropna(subset=['Gold_x', 'Gold_y'])
    temp = temp.drop(temp.index[-1])
    temp['Gold_diff'] = abs(temp.Gold_x - temp.Gold_y)
    q6 = temp.sort_values(by='Gold_diff', ascending=False)
    print("Question 6: \n", temp.loc[[q6.index[0]]])
    '''
    # another method
    temp1 = list(df['Gold_x'])
    temp2 = list(df['Gold_y'])
    temp1.pop()
    temp2.pop()
    for i in range(len(temp1)):
        if not isinstance(temp1[i],float):
            temp1[i] = re.sub(r',', '', temp1[i])
    temp1 = [float(x) for x in temp1]
    for i in range(len(temp2)):
        if not isinstance(temp2[i],float):
            temp2[i] = re.sub(r',', '', temp2[i])
    temp2 = [float(x) for x in temp2]
    L = []
    for i in range(len(temp1)):
        A = abs(temp1[i] - temp2[i])
        if A:
            L.append(A)
        else:
            L.append(max(temp1[i], temp2[i]))
    index = L.index(max(L))
    print("Question 6: \n", df['Unnamed: 0'][index])
    '''


def question_7(df):
    temp = df.drop(df.index[-1])
    q7 = temp.sort_values(by='Total.1', ascending=False)
    print("Question 7: \n", q7.head(5))
    print(q7.tail(5))
    return q7


def question_8(input):
    q8 = input.head(10)
    ax = q8.plot.barh(title='Question 8', x='Country', y=['Total_x', 'Total_y'], stacked=True)
    plt.show()


def question_9(input):
    q9 = input.set_index('Country')
    q9 = q9.rename({' United States (USA) [P] [Q] [R] [Z]':'United States', ' Australia (AUS) [AUS] [Z]':'Australia', ' Great Britain (GBR) [GBR] [Z]':'Great Britain', ' Japan (JPN)':'Japan', ' New Zealand (NZL) [NZL]':'New Zealand'}, axis='index')
    q9 = q9.loc[['United States', 'Australia', 'Great Britain', 'Japan', 'New Zealand']]
    ax2 = q9.plot.bar(title='Question 9', y=['Gold_y', 'Silver_y', 'Bronze_y'], rot=0)
    plt.show()


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500, 'display.max_columns', 500)
    #pd.set_option('display.height', 1000, 'display.width', 1000)
    df = question_1()
    question_2(df)
    question_3(df)
    question_4(df)
    question_5(df)
    question_6(df)
    q7_df = question_7(df)
    question_8(q7_df)
    question_9(q7_df)
