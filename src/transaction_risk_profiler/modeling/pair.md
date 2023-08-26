## Premise
You are a contract data scientist/consultant hired by a new e-commerce site to try to weed out fraudsters.  The company unfortunately does not have much data science expertise... so you must properly scope and present your solution to the manager before you embark on your analysis.  Also, you will need to build a sustainable software project that you can hand off to the companies engineers by deploying your model in the cloud.  Since others will potentially use/extend your code you **NEED** to properly encapsulate your code and leave plenty of comments.

## The Data
#### NOTE: This data is VERY sensitive!

It is located in `data/train_new.tar.gz`.

***Do not share this data with anyone or copy any of it off the mac minis! Do not include the data file in your pull request!***

You will be required to work on the project in the classroom.


### Deliverables (i.e. what you will be graded on)
* Scoping document (in Markdown)
* Code on private fork of repo on Github
    * proper functions/encapsulation
    * well commented
    * Model description document (see below)
* Flask app with well documented API
    * Needs to register your service at POST `/register`
    * Needs to accept input records on POST `/score` endpoint
* Web based front-end to enable quick triage of potential fraud
    * Triage importance of transactions (low risk, medium risk, high risk)
    * Extra: D3 based visualization of data/trend


### The "product" of fraud
Something that you will (need to) think about throughout this sprint is how the product of your client fits into the given technical process.  A few points to note about the case of fraud:

* Failures are not created equal
    * False positives decrease customer/user trust
    * False negatives cost money
        * Not all false negative cost the same amount of $$$
* Accessibility
    * Other (non-technical) people may need to interact with the model/machinery
    * Manual review

The fraud problem is actually semi-supervised in a way.  You do not use the model to declare a ground truth about fraud or not fraud, but simply to flag which transactions need further manual review.  We will essentially be building a triage model of what the most pressing (and costly) transaction we have seen.

## Day 1: Morning

### Step 1: EDA
Before we start building the model, let's start with some EDA.

#### [Deliverable]: Look at the data
Let's start by looking at the data.

1. Load the data in with pandas. Add a 'Fraud' column that contains True or False values depending on if the event is fraud. This is determined based on the `acct_type` field.

2. Check how many fraud and not fraud events you have.

3. Look at the features. Make note of ones you think will be particularly useful to you.

4. Do any data visualization that helps you understand the data.


#### [Deliverable]: Scoping the problem
Before you get cranking on your model, let's think of how to approach the problem.

1. Think of what preprocessing you might want to do. How will you build your feature matrix? What different ideas do you have?

2. What models do you want to try?

3. What metric will you use to determine success?


### Step 2: Building the Model

#### [Deliverable]: Comparing models
Now start building your potential models.

**Notes for writing code:**
* As you write your code, **always be committing** (ABC) to Github!
* Write **clean and modular code**.
* Include **comments** on every method.

*Make sure to get a working model first before you try all of your fancy ideas!*

1. If you have complicated ideas, implement the simplest one first! Get a baseline built so that you can compare more complicated models to that one.

2. Experiment with using different features in your feature matrix. Use different featurization techniques like stemming, lemmatization, tf-idf, part of speech tagging, etc.

3. Experiment with different models like SVM, Logistic Regression, Decision Trees, kNN, etc. You might end up with a final model that is a combination of multiple classification models.

4. Compare their results. Make sure to do good comparison and don't just use accuracy!

## Day 1: Afternoon

#### [Deliverable]: Model description and code
After all this experimentation, you should end up with a model you are happy with.

1. Create a file called `model.py` which builds the model based on the training data.

    * Feel free to use any library to get the job done.
    * Again, make sure your code is **clean**, **modular** and **well-commented**! The general rule of thumb: if you looked at your code in a couple months, would you be able to understand it?

2. In your pull request, describe your project findings including:
    * An overview of a chosen “optimal” modeling technique, with:
        * process flow
        * preprocessing
        * accuracy metrics selected
        * validation and testing methodology
        * parameter tuning involved in generating the model
        * further steps you might have taken if you were to continue the project.


#### [Deliverable]: Pickled model

1. Use `cPickle` to serialize your trained model and store it in a file. This is going to allow you to use the model without retraining it for every prediction, which would be ridiculous.

### Step 3: Prediction script

Take a few raw examples and store them in json or csv format in a file called `test_script_examples` or the like.


#### [Deliverable]: Prediction script

1. Write a script `predict.py` that reads in a single example from `test_script_examples`, vectorizes it, unpickles the model, predicts the label, and outputs the label probability (print to standard out is fine).

    This script will serve as a sort of conceptual and code bridge to the web app you're about to build.

    Each time you run the script, it will predict on one example, just like a web app request. You may be thinking that unpickling the model every time is quite inefficient and you'd be right; we'll remove that inefficiency in the web app.


### Step 4: Database

#### [Deliverable]: Prediction script backed by a database

We want to store each prediction the model makes on new examples, which means we'll need a database.

1. Set up a Postgres or MongoDB database that will store each example that the script runs on. You should create a database schema that reflects the form of the raw example data and add a column for the predicted probability of fraud.

2. Write a function in your script that takes the example data and the prediction as arguments and inserts the data into the database.

    Now, each time you run your script, one row should be added to the `predictions` table with a predicted probability of fraud.

## Day 2: Morning

### Step 5: Web App

#### [Deliverable]: Hello World app

1. A request in your browser to `/hello` should display "Hello, World!". Set up a Flask app with a route `GET /hello` to do so.

    Use this [tutorial](http://blog.luisrei.com/articles/flaskrest.html) to help.

#### [Deliverable]: Fraud scoring service

1. Set up a route `POST /score` and have it execute the logic in your prediction script. You should import the script as a module and call functions defined therein.

    There are two things we'll do to make this all more efficient:

    1. We only want to unpickle the model once
    2. We only want to connect to the database once.

    Do both in a `if __name__ == '__main__':` block before you call `app.run()` and you can refer to these top-level global variables from within the function. This may require some re-architecting of your prediction module.

    The individual example will no longer be coming from a local file, but instead will come in the body of the POST request as JSON. You can use `request.data` or `request.json` to access that data. You'll still need to vectorize it, predict, and store the example and prediction in the database.

    You can test out this route by, in a separate script, sending a POST request to /score with a single example in JSON form using the `requests` Python package.


### Step 6: Get "live" data

We've set up a service for you that will ping your server with "live" data so that you can see that it's really working.

To use this service, you will need to make a POST request to `ourcomputersip/register` with your IP and port. We'll announce what the ip address of the service machine is in class.

Here is the code to register your machine:

```python
import requests
reg_url = 'http://<to be filled in>/register:5000'
requests.post(reg_url, data={'ip': my_ip, 'port': my_port})
```

You can get your computer's IP with the `socket` module as can be seen [here](http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib). The port is your choice.

1. Write a register function which makes the necessary post request. This function should be called once each time your Flask app is run, in the main block.

**Make sure your app is adding the examples to the database with predicted fraud probabilities.**

## Day 2: Afternoon

### Step 7: Dashboard

#### [Deliverable]: Web front end to present results

We want to present potentially fraudulent transactions with their probability scores from our model. The transactions should be segmented into 3 groups: low risk, medium risk, or high risk (based on the probabilities).

* Add route in Flask app for dashboard
* Read data from postgres
* Return HTML with the data
    * To generate the HTML from the json data from the database, either just use simple string concatenation or Jinja2 templates.


### Step 8: Deploy!

Use the [aws sprint](https://github.com/zipfian/high_performance_python/blob/master/individual.md) as your guide if you need one.

**The data stream is not available to you on AWS. Change your web app to display only the predictions of the test set. You can also change your app such that the results are not written to a database.**

* Set up AWS instance
* Set up environment on your EC2 instance
* Push your code to github
* SSH into the instance and clone your repo
* Run Flask app on instance (make sure you update the register code with your updated ip and port)
* Make it work (debug, debug, debug)
* Profits!


### Extra

* Make your dashboard interactive. Allow a dashboard user to clear or flag fraud events. Come up with other features that might be useful.

* Create a D3 visualization for your web based frontend.  You might want to visualize any number of metrics/data.  Use your creativity to create something that makes sense for a end user in terms of what data you present.
