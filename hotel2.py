# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#00CED1;font-family:cursive;color:#FFF8DC;font-size:200%;text-align:center;border-radius:25px;"> 🚀 Let's Dive In! 🌟🌟 </p>
#
#
#

# %% [markdown]
# <center><iframe src="https://gifer.com/embed/2Yhi" width=480 height=366.933 frameBorder="0" allowFullScreen></iframe><p><a href="https://gifer.com"></a></p></center>

# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#008080;font-family:Helvetica, Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px;padding:10px;"> 📝 Introduction 📖 </p>
#

# %% [markdown]
# <p style="font-family: 'Segoe UI'; font-size: 20px; line-height: 1.6; color: #800080;">
#     In the past few years, both the City Hotel and Resort Hotel have experienced significant increases in their cancellation rates. As a result, both hotels are currently facing a range of challenges, such as reduced revenue and underutilized hotel rooms. Therefore, the top priority for both hotels is to reduce their cancellation rates, which will enhance their efficiency in generating revenue. This report focuses on the analysis of hotel booking cancellations and other factors that do not directly impact their business and annual revenue generation.
# </p>
#

# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#6A5ACD;font-family:Tahoma, Geneva, sans-serif;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px;padding:10px;"> 📊 Learn About Data 📈 </p>
#

# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:16px;color:#white;">This dataset contains 119390 observations for a City Hotel and a Resort Hotel. Each observation represents a hotel booking between the 1st of July 2015 and 31st of August 2017, including booking that effectively arrived and booking that were canceled.</p>
#
# Columns:
#
#
# * **hotel:** One of the hotels is a resort hotel and the other is a city hotel.
# * **is_canceled	lead_time:** Value indicating if the booking was canceled (1) or not (0).
# * **arrival_date_year:** Year of arrival date.
# * **arrival_date_month:** Month of arrival date with 12 categories: “January” to “December”.
# * **arrival_date_week_number:** Week number of the arrival date.
# * **arrival_date_day_of_month:** Day of the month of the arrival date.
# * **stays_in_weekend_nights:** Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel.
# * **stays_in_week_nights:** Number of week nights (Monday to Friday) the guest stayed.
# * **adults:** Number of adults
# * **children:**	Number of Childern
# * **babies:** Number of Babies
# * **meal:** BB – Bed & Breakfast
# * **country:** Country of origin.
# * **market_segment:** Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”
# * **distribution_channel:** Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”
# * **is_repeated_guest:** Value indicating if the booking name was from a repeated guest (1) or not (0)
# * **previous_cancellations:** Number of previous bookings that were cancelled by the customer prior to the current booking
# * **previous_bookings_not_canceled:** Number of previous bookings not cancelled by the customer prior to the current booking
# * **reserved_room_type:** Code of room type reserved. Code is presented instead of designation for anonymity reasons.
# * **assigned_room_type:** Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons
# * **booking_changes:** Number of changes/amendments made to the booking.
# * **deposit_type:** No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.
# * **agent:** ID of the travel agency that made the booking
# * **company:** ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons
# * **days_in_waiting_list:**  Number of days the booking was in the waiting list before it was confirmed to the customer
# * **customer_type:** Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking
# * **adr:**  Average Daily Rate (Calculated by dividing the sum of all lodging transactions by the total number of staying nights)
# * **required_car_parking_spaces:**  Number of car parking spaces required by the customer
# * **total_of_special_requests:**  Number of special requests made by the customer (e.g. twin bed or high floor)
# * **reservation_status:** Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why
# * **reservation_status_date:** Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel
# * **name:** Name of the Guest (Not Real)
# * **email:**  Email (Not Real)
# * **phone-number:** Phone number (not real)
#

# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#20B2AA;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px;padding:10px;"> 🔄 Life Cycle Of Machine Learning Project 🤖 </p>
#
#    <ul style="font-size: 18px; font-family: 'Segoe UI';">
#         <li><strong>Understanding the Problem Statement</strong></li>
#         <li><strong>Data Checks to Perform</strong></li>
#         <li><strong>Exploratory Data Analysis</strong></li>
#         <li><strong>Data Pre-Processing</strong></li>
#         <li><strong>Model Training</strong></li>
#         <li><strong>Choose Best Model</strong></li>
#     </ul>
# </div>

# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#FFA07A;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> 1️) Problem Statement 🎯 </p>
#

# %% [markdown]
# In the past few years, both the City Hotel and Resort Hotel have experienced significant increases in their cancellation rates. As a result, both hotels are currently facing a range of challenges, such as reduced revenue and underutilized hotel rooms. Therefore, the top priority for both hotels is to reduce their cancellation rates, which will enhance their efficiency in generating revenue.
# This report focuses on the analysis of hotel booking cancellations and other factors that do not directly impact their business and annual revenue generation.
#

# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <p style="background-color:#20B2AA;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> Assumptions 🤔 </p>
# </div>
#
# 1.	Between 2015 and 2017, no significant events or unexpected incidents had a substantial impact on the data being considered.
# 2.	The information remains up-to-date and can be effectively utilized to analyze a hotel's potential strategies.
# 3.	There are no unforeseen drawbacks to the hotel's adoption of any recommended approach.
# 4.	The suggested solutions are not currently being implemented by the hotels.
# 5.	The most significant factor influencing revenue generation is the occurrence of booking cancellations.
# 6.	Cancellations result in unoccupied rooms for the originally booked duration.
# 7.	Clients typically make hotel reservations in the same year they subsequently cancel them.
#
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <p style="background-color:#6A5ACD;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> Research Question ❓ </p>
# </div>
#
# 1.	What are the variables that effect hotel reservation cancellations?
# 2.	How can we make hotel reservations cancellation better?
# 3.	How all hotels be assisted in making pricing and promotional decisions?
#
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <p style="background-color:#FF6347;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> Hypothesis 🧪 </p>
# </div>
#
# 1.	More cancellations occur when prices are higher.
# 2.	When there is a longer waiting list. Customers tend to cancel more frequently.
# 3.	The majority of clients are coming from offline travel agents to make their reservations.
#

# %% [markdown]
# <font size="+2" color=red ><b>Please Upvote my kernel if you like my work.</b></font>
#

# %% [markdown]
# # <a id='Libraries'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#008080;font-family:Helvetica, Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px;padding:10px;"> 📚 Importing Libraries 📦 </p>
#

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#FF6347;font-family:Verdana, sans-serif;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px;padding:10px;"> 📊 Loading the Dataset 📁 </p>
#

# %%
df = pd.read_csv('/kaggle/input/hotel-booking/hotel_booking.csv')

# %% [markdown]
# # <div style="color: blue; display:inline-block; border-radius: 5px; background-color: #F0E68C; font-family: 'Nexa', sans-serif; overflow: hidden;"><p style="padding: 15px; color: blue; overflow: hidden; font-size: 100%; letter-spacing: 0.5px; margin: 0; width: 750px;"><b>2) Data Checks to Perform</b></p>
# </div>
#
#   <ul style="font-family: 'Segoe UI'; font-size: 20px; margin-left: 20px;">
#     <li>Check Missing values</li>
#     <li>Check Duplicates</li>
#     <li>Check data type</li>
#     <li>Check the number of unique values of each column</li>
#     <li>Check statistics of the dataset</li>
#     <li>Check various categories present in the different categorical columns</li>
#   </ul>
#

# %% [markdown]
# # <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#4682B4;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> 3️) Exploratory Data Analysis 📊 </p>
#
#

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df.info()

# %%
df['reservation_status_date']= pd.to_datetime(df['reservation_status_date'])

# %%
df.info()

# %%
df.info()

# %%
df.describe(include = 'object')

# %%
for col in df.describe(include='object').columns:
    print(col)
    print(df[col].unique())
    print('-'*50)

# %% [markdown]
# # <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
# <p style="background-color:#FF8C00;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> 4) Data Pre-processing 🛠️ </p>
#

# %% [markdown]
# ## Handling Missing Values

# %%
df.isnull().sum()

# %%
df.drop(['company','agent'], axis =1, inplace = True)
df.dropna(inplace =True)

# %%
df.isnull().sum()

# %%
df.describe()

# %%
df = df[df['adr']<5000]

# %%
df.describe()

# %% [markdown]
# # <a id='top'></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <p style="background-color:#FF8C00;font-family:Arial, sans-serif;color:#FFFFFF;font-size:150%;text-align:left;border-radius:10px;padding:10px;"> 5) Data Analysis and Visualization 🛠️ </p>
# </div>
#

# %%
cancelled_perc = df['is_canceled'].value_counts(normalize=True)
print(cancelled_perc)
colors = sns.color_palette(["#FFEC9E", "#FFBB70"])

plt.figure(figsize = (5,3),facecolor="#C38154")
plt.title('Reservation Status')
plt.bar(['Not cancelled' , 'Cancelled'],df['is_canceled'].value_counts(), edgecolor = 'k', width = 0.6,color=colors)
plt.show()

# %%
# Calculate cancellation percentages
cancelled_perc = df['is_canceled'].value_counts(normalize=True)

# Define color palette
colors = sns.color_palette(["#FFEC9E", "#FFBB70"])

# Plot
plt.figure(figsize=(8, 6), facecolor='#C38154')  # Set background color to light gray
plt.title('Reservation Status', fontsize=16, color='Black')
plt.bar(['Not Cancelled', 'Cancelled'], df['is_canceled'].value_counts(), edgecolor='black', width=0.6, color=colors)
plt.xlabel('Reservation Status', fontsize=14, color='Black')
plt.ylabel('Count', fontsize=14, color='Black')
plt.xticks(fontsize=12, color='Black')
plt.yticks(fontsize=12, color='Black')
plt.tight_layout()
plt.savefig('reservation_status_plot.png', bbox_inches='tight', transparent=True)
plt.show()


# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">The provided bar graph illustrates the cancellation and non-cancellation percentages for reservations. It is evident that a substantial portion of reservations remains unaffected by cancellations. Notably, 37% of clients have chosen to cancel their reservations, and this has a noteworthy impact on the hotels' revenue.</p>

# %%
# Set the background color of the figure
plt.figure(figsize=(8, 4), facecolor='#C38154')

# Create the countplot
ax1 = sns.countplot(x='hotel', hue='is_canceled', data=df, palette="Set2")

# Customize legend location
legend_labels, _ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1, 1))

# Set plot title and axis labels
plt.title('Reservation status in different hotels', size=20, color='Black')
plt.xlabel('Hotel',color='Black')
plt.ylabel('Number of Reservations', color='Black')

# Customize legend labels
plt.legend(['Not Cancelled', 'Cancelled'])

# Show the plot
plt.show()


# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">In comparison to resort hotels, city hotels have more bookings. Its possible that resort hotels are more expensive that those in cities.</p>

# %%
#checking the graph cancelation rate in percentage for Resort Hotel
resort_hotel= df[df['hotel']=='Resort Hotel']
resort_hotel['is_canceled'].value_counts(normalize = True)


# %%
# checking the Above graph Cancellation rate in percentage for City Hotel
City_Hotel = df[df['hotel']== 'City Hotel']
City_Hotel['is_canceled'].value_counts(normalize =True)


# %%
resort_hotel = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
City_Hotel = City_Hotel.groupby('reservation_status_date')[['adr']].mean()

# %%
plt.figure(figsize=(20,8), facecolor='#C38154')
plt.title('Average Daily Rate in City and Resort Hotel', fontsize=30)
plt.plot(resort_hotel.index,resort_hotel['adr'],label = 'Resort Hotel')
plt.plot(City_Hotel.index,City_Hotel['adr'],label = 'City Hotel')
plt.legend(fontsize=20)
plt.show()

# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:16px;color:#white;">The line graph above shows that, on certain days the average daily rate for a city hotel is less than that of a resort hotel and on other days, it is even less.<br>It goes without saying that weekends and holidays may see a rise in resort hotel rates.</p>

# %%
df['month']=df['reservation_status_date'].dt.month
plt.figure(figsize=(16,8), facecolor='#C38154')
ax1 = sns.countplot(x='month', hue='is_canceled', data= df, palette = 'Set2')
legend_lebels,_ = ax1.get_legend_handles_labels()
plt.title('Reservation Status Per Month', size = 20)
plt.xlabel('month')
plt.ylabel('Number of Reservation')
plt.legend(['Not Canceled','Canceld'])
plt.show()

# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">We've created a grouped bar graph to examine the months with the highest and lowest reservation levels based on their status. It's evident that the month of August stands out, having the highest numbers of both confirmed and canceled reservations. In contrast, January has the fewest confirmed reservations but the highest number of canceled reservations.</p>

# %%
df['adr']

# %%
plt.figure(figsize = (15,8))
plt.title('ADR per Month', fontsize = 30)
# data =df[df['is_canceled'] == 1].groupby('month')['adr'].sum().reset_index()

data = df[df['is_canceled'] == 1].groupby('month')['adr'].sum().reset_index()
sns.barplot(x='month', y='adr', data = data )
plt.legend(fontsize = 20)
plt.show()

# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">This bar graph illustrates that cancellations are most frequent when prices are at their highest and least common when prices are at their lowest. Consequently, the price of accommodation appears to be the primary factor influencing cancellations.<br>
# Now, let's examine which country experiences the highest number of canceled reservations. Portugal stands out as the top country with the highest number of cancellations.
# </p>

# %%
cancelled_data= df[df['is_canceled']==1]
top_10_country = cancelled_data['country'].value_counts()[:10]
# plt.figure(figsize=(8,8), facecolor='#C38154')
# plt.title('Top 10 countries with reservation canceled',color="black")
# plt.pie(top_10_country, autopct ='%.2f', labels = top_10_country.index)
# plt.show()

# Custom colors for the pie chart
custom_colors = ['#FF6347', '#4682B4', '#7FFF00', '#FFD700', '#87CEEB', '#FFA07A', '#6A5ACD', '#FF69B4', '#40E0D0', '#DAA520']

plt.figure(figsize=(8, 8), facecolor='#C38154')  # Set background color to a light brown
plt.title('Top 10 countries with reservation canceled', color="black")
plt.pie(top_10_country, autopct='%.2f', labels=top_10_country.index, colors=custom_colors)
plt.show()


# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">Let's analyze the sources from which guests are making hotel reservations, including Direct, Groups, Online Travel Agencies, and Offline Travel Agents. <br>Approximately 46% of clients make reservations through online travel agencies, while 27% come through group bookings. <br>A mere 4% of clients choose to book hotels directly by visiting them in person and making reservations.</p>

# %%
df['market_segment'].value_counts()

# %%
df['market_segment'].value_counts(normalize=True)


# %%
cancelled_data['market_segment'].value_counts(normalize=True)

# %%
cancelled_df_adr= cancelled_data.groupby('reservation_status_date')[['adr']].mean()
cancelled_df_adr.reset_index(inplace=True)
cancelled_df_adr.sort_values('reservation_status_date', inplace=True)

not_cancelled_data=df[df['is_canceled']==0]
not_cancelled_df_adr= not_cancelled_data.groupby('reservation_status_date')[['adr']].mean()
not_cancelled_df_adr.reset_index(inplace=True)
not_cancelled_df_adr.sort_values('reservation_status_date', inplace=True)

plt.figure(figsize=(20,6), facecolor='#C38154')
plt.title('Average Daily Rate', color="Black")
plt.plot(not_cancelled_df_adr['reservation_status_date'],not_cancelled_df_adr['adr'], label='not cancelled')
plt.plot(cancelled_df_adr['reservation_status_date'],cancelled_df_adr['adr'], label = 'cancelled')
plt.legend()
plt.show()

# %%
cancelled_df_adr = cancelled_df_adr[(cancelled_df_adr['reservation_status_date']>'2016') & (cancelled_df_adr['reservation_status_date']<'2017-09')]

not_cancelled_df_adr = not_cancelled_df_adr[(not_cancelled_df_adr['reservation_status_date']>'2016') & (not_cancelled_df_adr['reservation_status_date']<'2017-09')]

# %%
plt.figure(figsize=(20,6), facecolor='#C38154')
plt.title('Average Daily Rate', fontsize = 40)
plt.plot(not_cancelled_df_adr['reservation_status_date'],not_cancelled_df_adr['adr'], label='not cancelled')
plt.plot(cancelled_df_adr['reservation_status_date'],cancelled_df_adr['adr'], label = 'cancelled')
plt.legend(fontsize = 20)
plt.show()

# %% [markdown]
# ><p style="font-family:Arial, sans-serif;font-size:20px;color:#white;">As seen in the graph, reservations are canceled when the average daily rate is higher than when it is not canceled. <br>It clearly proves all the above analysis that the higher price leads to higher cancellation.</P>

# %% [markdown]
# <a id="top"></a>
# <div class="list-group" id="list-tab" role="tablist">
#   <p
#     style="
#       background-color: #800080;
#       font-family: Arial, sans-serif;
#       color: #ffffff;
#       font-size: 150%;
#       text-align: left;
#       border-radius: 10px;
#       padding: 10px;
#     "
#   >
#     6) Final Result 🏁
#   </p>
# </div>
#
# <p style="font-family:Arial, sans-serif;font-size:30px;color:Orange;"><b>Suggestions:</b></p>
#
# <a id='top'></a>
# <ol style="font-family:Arial, sans-serif;font-size:20px;color:#white;">
#   <li>Increasing prices are associated with a higher rate of cancellations. To mitigate reservation cancellations, hotels could refine their pricing strategies by offering reduced rates for specific locations and providing discounts to customers.</li>
#   <li>The resort hotel experiences a higher ratio of cancellations compared to the city hotels. Therefore, hotels should consider offering competitive room price discounts on weekends and holidays.</li>
#   <li>During the month of January, hotels can launch marketing campaigns with attractive offers to boost their revenue, especially since cancellations tend to peak during this period.</li>
#   <li>Enhancing the quality of hotels and their services, particularly in Portugal, can be an effective approach.
#

# %% [markdown]
# <div style="background-color:#87CEEB; padding:20px; border-radius:10px;">
#   <h2 style="color:Green; font-family:Arial, sans-serif; text-align:center;font-size:24px;">Notes 😃😃😃😃</h2>
#   <ul style="color:#000000; font-family:Arial, sans-serif; font-size:20px;">
#     <li>Thank you for reading my analysis and regression. 😃😃😃😃</li>
#     <li>If you have any questions or advice, please write in the comments. ❤️❤️❤️❤️</li>
#     <li>If anyone has a model with a higher percentage, please let me know. 🤝🤝🤝</li>
#   </ul>
# </div>
#
# <div style="background-color:#FFA07A; padding:20px; border-radius:10px;">
#   <h2 style="color:#FFFFFF; font-family:Arial, sans-serif; text-align:center;">Vote ❤️😃</h2>
#   <p style="color:#000000; font-family:Arial, sans-serif; text-align:center; font-size:20px;"><b>If you enjoyed this analysis, an Upvote would be the cherry on top!</b> 🍒👍</p>
# </div>
#
# <div style="background-color:#90EE90; padding:20px; border-radius:10px;">
#   <h2 style="color:#FFFFFF; font-family:Arial, sans-serif; text-align:center;font-size:20px;">The End 🤝🎉🤝🎉</h2>
# </div>
#


