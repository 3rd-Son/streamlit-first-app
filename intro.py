import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
import time


# Set title
st.title("MY FIRST STREAMLIT APP")

from PIL import Image

st.subheader("`I AM VIC3SAX`")

image = Image.open("/Users/appleplay/Downloads/IMG_9728.jpg")
st.image(image, use_column_width=True)

st.write("I am a Full Stack Data Scientist with strong knowledge of python and SQL")

st.markdown(" *This is a markdown text*")

st.success("Congrats you ran the app successfully")
st.info("This is an informative message")
st.warning("This is a warning message")
st.error(
    "oops you ran into an error, you need to `rerun` the app or `install` the app again"
)

st.help(len)

dataframe = np.random.rand(10, 200)
st.dataframe(dataframe)

st.text("****" * 100)

df = pd.DataFrame(np.random.rand(10, 20), columns=("col %d" % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

# Plots
st.text("---" * 100)

chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

st.area_chart(chart_data)
st.bar_chart(chart_data)
st.set_option("deprecation.showPyplotGlobalUse", False)

# Matplotlib plot
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)

st.pyplot()

st.text("----" * 100)

# Plotly plot
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) - 2

hist_data = [x1, x2, x3]
group_lables = ["Group 1", "Group 2", "Group 3"]

fig = ff.create_distplot(hist_data, group_lables, bin_size=[0.2, 0.25, 0.5])
st.plotly_chart(fig, use_container_width=True)


st.text("----" * 100)

df = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=["latitude", "longitude"],
)
st.map(df)

st.text("****" * 100)

if st.button("Say hello"):
    st.write("hello")

elif st.button("Touch me"):
    st.write("You are welcome")

else:
    st.button("Why are you here?")

st.text("----" * 100)

genre = st.radio("What is your favorite genre?", ("Comedy", "Drama", "Documentary"))

if genre == "Comedy":
    st.write("Oh so na comedy you dy watch")
elif genre == "Drama":
    st.write("Ndi drama ndewo")
else:
    st.write(
        "I see say na matter wey no concern you you dy like put mouth for nwoke documentary (okay na)"
    )

st.text("---" * 100)
# select box

option = st.selectbox("How was your night", ("Fantastic", "Awesome", "so-so"))
st.write("You said your night was", option)

st.text("---" * 100)

option = st.multiselect(
    "How was your night, `select multipe choice`", ("Fantastic", "Awesome", "so-so")
)
st.write("You said your night was", option)

st.text("---" * 100)

age = st.slider("How old are you?", 0, 150, 10)
st.write(f"Your age is {age}")

st.text("---" * 100)

values = st.slider("Select a range?", 0, 100, (10, 20))
st.write(f"Your age is {values}")

number = st.number_input("Input no")
st.write("The number is ", number)

st.text("--" * 50)

upload_file = st.file_uploader("Choose a CSV file", type="csv")

if upload_file is not None:
    data = pd.read_csv(upload_file)
    st.write(data)
    st.success("Sucessfully uploaded")
else:
    st.error("The file you uploaded is empty, please upload a valid CSV file")


color = st.color_picker("Pick your color", "#000f90")
st.write("this is your color", color)

side_bar = st.sidebar.selectbox(
    "What is your favorite course?",
    (
        "DA",
        "DS",
        "DE",
        "Im not sure",
    ),
)

# to display a progress bar

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)
with st.spinner("wait for it"):
    time.sleep(5)
