/**
 * User: Parth
 * Date: 13/3/12
 * Time: 12:03 AM
 */

//error_reporting(E_ERROR);

function consolelog(txt) {

  $('#console').append(txt + "<br>");
}

var _RAND_MAX = 32767;

var HI = 0.9;

var LO = 0.1;

clone = function (x) {
  var newObj = (x instanceof Array) ? [] : {};
  for (i in x) {
    if (i == 'clone') continue;
    if (x[i] && typeof x[i] == "object") {
      newObj[i] = clone(x[i]);
    } else newObj[i] = x[i]
  }
  return newObj;
};

makeInt = function (x) {
  if (!(x instanceof Array) && !(x instanceof Object))return parseInt(x);
  var y = clone(x);
  for (i in y) {
    y[i] = makeInt(y[i]);
  }
  return y;
}

function mt_rand(min, max) {
  // Returns a random number from Mersenne Twister
  //
  // version: 1109.2015
  // discuss at: http://phpjs.org/functions/mt_rand
  // +   original by: Onno Marsman
  // *     example 1: mt_rand(1, 1);
  // *     returns 1: 1
  var argc = arguments.length;
  if (argc === 0) {
    min = 0;
    max = 2147483647;
  } else if (argc === 1) {
    throw new Error('Warning: mt_rand() expects exactly 2 parameters, 1 given');
  }
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function abs(mixed_number) {
  // Return the absolute value of the number
  //
  // version: 1109.2015
  // discuss at: http://phpjs.org/functions/abs
  // +   original by: Waldo Malqui Silva
  // +   improved by: Karol Kowalski
  // +   improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
  // +   improved by: Jonas Raoni Soares Silva (http://www.jsfromhell.com)
  // *     example 1: abs(4.2);
  // *     returns 1: 4.2
  // *     example 2: abs(-4.2);
  // *     returns 2: 4.2
  // *     example 3: abs(-5);
  // *     returns 3: 5
  // *     example 4: abs('_argos');
  // *     returns 4: 0
  return Math.abs(mixed_number) || 0;
}

function exp(arg) {
  // Returns e raised to the power of the number
  //
  // version: 1109.2015
  // discuss at: http://phpjs.org/functions/exp
  // +   original by: Onno Marsman
  // *     example 1: exp(0.3);
  // *     returns 1: 1.3498588075760032
  return Math.exp(arg);
}

var BackPropagationScale = {


  initialize:function (numLayers, layersSize, beta, alpha, minX, maxX) {
    /* Output of each neuron */
    this.output = new Array(numLayers);
    /* Last calcualted output value */
    this.vectorOutput = new Array;
    /* delta error value for each neuron */
    this.delta = new Array(numLayers);
    /* Array of weights for each neuron */
    this.weight = new Array(numLayers);

    /* Storage for weight-change made in previous epoch (three-dimensional array) */
    this.prevDwt = new Array(numLayers);

    /* Data */
    this.data = null;
    /* Test Data */
    this.testData = null;
    /* N lines of Data */
    this.NumPattern = null;
    /* N columns in Data */
    this.NumInput = null;

    /* Stores ann scale calculated parameters */
    this.normalizeMax = null;
    this.normalizeMin = null;

    /* Holds all output data in one array */
    this.output_vector = null;


    /* Learning rate */
    this.alpha = alpha;
    /* Momentum */
    this.beta = beta;
    /* Minimum value in data set */
    this.minX = minX;
    /* Maximum value in data set */
    this.maxX = maxX;


    /* Num of layers in the net, including input layer */
    this.numLayers = numLayers;
    /* Array num elments containing size for each layer */
    this.layersSize = layersSize;

    //	seed and assign random weights
    consolelog("Initialise Weights");
    for (i = 1; i < this.numLayers; i++) {
      this.weight[i] = new Array(this.layersSize[i]);
      for (j = 0; j < this.layersSize[i]; j++) {
        this.weight[i][j] = new Array(this.layersSize[i - 1] + 1)
        for (k = 0; k < this.layersSize[i - 1] + 1; k++) {
          this.weight[i][j][k] = this.rando();
        }
        // bias in the last neuron
        this.weight[i][j][this.layersSize[i - 1]] = -1;
      }
    }

    //	initialize previous weights to 0 for first iteration
    for (i = 1; i < this.numLayers; i++) {
      this.prevDwt[i] = new Array(this.layersSize[i]);
      for (j = 0; j < this.layersSize[i]; j++) {
        this.prevDwt[i][j] = new Array(this.layersSize[i - 1] + 1);
        for (k = 0; k < this.layersSize[i - 1] + 1; k++) {
          this.prevDwt[i][j][k] = 0.0;
        }
      }
    }
  },
  rando:function () {

    var randValue = LO + (HI - LO) * mt_rand(0, _RAND_MAX) / _RAND_MAX;
    return randValue;//32767
  },

  /* ---	sigmoid function */
  sigmoid:function (inputSource) {
    return (1.0 / (1.0 + exp(-inputSource)));
  },

  /* --- mean square error */
  mse:function (target) {
    var mse = 0;
    for (var i = 0; i < this.layersSize[this.numLayers - 1]; i++) {
      mse += (target - this.output[this.numLayers - 1][i]) *
        (target - this.output[this.numLayers - 1][i]);
    }
    return mse / 2;
  },

  /* ---	returns i'th outputput of the net */
  Out:function (i) {

    return this.output[this.numLayers - 1][i];
  },

  /* ---
     * Feed forward one set of input
     * to update the output values for each neuron. This function takes the input
     * to the net and finds the output of each neuron
     */
  ffwd:function (inputSource) {
    //consolelog("Doing ffwd for: " + inputSource);
    sum = 0.0;
    //	assign content to input layer
    this.output = new Array(this.numLayers);
    this.output[0] = new Array(this.layersSize[0]);
    for (i = 0; i < this.layersSize[0]; i++) {
      this.output[0][i] = inputSource[i];  // outputput_from_neuron(i,j) Jth neuron in Ith Layer
    }


    //	assign output (activation) value to each neuron usng sigmoid func
    for (var i = 1; i < this.numLayers; i++)                    // For each layer
    {
      this.output[i] = new Array(this.layersSize[i]);
      for (j = 0; j < this.layersSize[i]; j++)                // For each neuron in current layer
      {
        var sum = 0.0;
        for (k = 0; k < this.layersSize[i - 1]; k++)            // For each input from each neuron in preceeding layer
        {
          sum += this.output[i - 1][k] * this.weight[i][j][k];	// Apply weight to inputs and add to sum
        }
        // Apply bias
        sum += this.weight[i][j][this.layersSize[i - 1]];
        // Apply sigmoid function
        this.output[i][j] = this.sigmoid(sum);
      }
    }
    //consolelog(this.output);
  },

  /* ---	Backpropagate errors from outputput	layer back till the first hidden layer */
  bpgt:function (inputSource, target) {
    //consolelog("Bpgt target: " + target);
    /* ---	Update the output values for each neuron */
    this.ffwd(inputSource);

    ///////////////////////////////////////////////
    /// FIND DELTA FOR OUPUT LAYER (Last Layer) ///
    ///////////////////////////////////////////////


    this.delta[this.numLayers - 1] = new Array(this.layersSize[this.numLayers - 1]);
    for (var k = 0; k < this.layersSize[this.numLayers - 1]; k++) {
      //this.delta[this.numLayers - 1] = new Array(this.layersSize[this.numLayers - 1]);
      this.delta[this.numLayers - 1][k] = this.beta * this.output[this.numLayers - 1][k] *
        (1 - this.output[this.numLayers - 1][k]) *
        (target - this.output[this.numLayers - 1][k]);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    /// FIND DELTA FOR HIDDEN LAYERS (From Last Hidden Layer BACKWARDS To First Hidden Layer) ///
    /////////////////////////////////////////////////////////////////////////////////////////////

    for (i = this.numLayers - 2; i > 0; i--) {
      this.delta[i] = new Array(this.layersSize[i]);
      for (j = 0; j < this.layersSize[i]; j++) {
        var sum = 0.0;
        for (k = 0; k < this.layersSize[i + 1]; k++) {
          sum += this.delta[i + 1][k] * this.weight[i + 1][k][j];
        }
        this.delta[i][j] = this.beta * this.output[i][j] * (1 - this.output[i][j]) * sum;
      }
    }

    ////////////////////////
    /// MOMENTUM (Alpha) ///
    ////////////////////////

    /*for (i = 1; i < this.numLayers; i++) {
     for (j = 0; j < this.layersSize[i]; j++) {
     for (k = 0; k < this.layersSize[i - 1]; k++) {
     this.weight[i][j][k] += this.alpha * this.prevDwt[i][j][k];
     }
     this.weight[i][j][this.layersSize[i - 1]] += this.alpha * this.prevDwt[i][j][this.layersSize[i - 1]];
     }
     }*/

    ///////////////////////////////////////////////
    /// ADJUST WEIGHTS (Using Steepest Descent) ///
    ///////////////////////////////////////////////

    for (i = 1; i < this.numLayers; i++) {
      for (j = 0; j < this.layersSize[i]; j++) {
        for (k = 0; k < this.layersSize[i - 1]; k++) {
          this.prevDwt[i][j][k] = this.beta * this.delta[i][j] * this.output[i - 1][k];
          this.weight[i][j][k] += this.prevDwt[i][j][k];
        }
        /* --- Apply the corrections */
        this.prevDwt[i][j][this.layersSize[i - 1]] = this.beta * this.delta[i][j];
        this.weight[i][j][this.layersSize[i - 1]] += this.prevDwt[i][j][this.layersSize[i - 1]];
      }
    }
  },


  ///////////////////////////////
  /// SCALING FUNCTIONS BLOCK ///
  ///////////////////////////////
  /* --- Set scaling parameters */
  setScaleOutput:function (data) {
    if (!(data instanceof Array))return;
    oldMin = data[0][0];
    oldMax = oldMin;
    var numElem = data[0].length;

    /* --- First calcualte minimum and maximum */
    for (i = 0; i < data.length; i++) {
      if (!(data[i] instanceof Array))continue;
      for (j = 1; j < numElem; j++) {
        data[i][j] = data[i][j];
        // Min
        if (oldMin > data[i][j]) {
          oldMin = data[i][j];
        }
        // Max
        if (oldMax < data[i][j]) {
          oldMax = data[i][j];
        }
      }
      this.normalizeMin[i] = oldMin;
      this.normalizeMax[i] = oldMax;
    }

  },

  /* --- Scale input data to range before feeding it to the network */
    /*
     x - Min
     t = (HI -LO) * (---------) + LO
     Max-Min
     */
  scale:function (data) {
    this.setScaleOutput(data);
    var numElem = data[0].length;
    temp = 0.0;
    for (i = 0; i < data.length; i++) {
      for (j = 0; j < numElem; j++) {
        //consolelog('(' + data[i][j] + ' - ' + this.normalizeMin[i] + ') / (' + this.normalizeMax[i] + ' - ' + this.normalizeMin[i] + ') + ' + LO);
        temp = (HI - LO) * ((data[i][j] - this.normalizeMin[i]) / (this.normalizeMax[i] - this.normalizeMin[i])) + LO;
        //consolelog("\t" + i + ", " + j + ": " + temp);
        if (!isNaN(temp))
          data[i][j] = temp;
        else
          data[i][j] = 0;
      }
    }
    return data;
  },

  /* --- Unscale output data to original range */
    /*
     x - LO
     t = (Max-Min) * (---------) + Min
     HI-LO
     */
  unscaleOutput:function (output_vector) {

    temp = 0.0;

    var unscaledVector = new Array(this.NumPattern);
    for (i = 0; i < output_vector.length; i++) {

      temp = (this.normalizeMax[i] - this.normalizeMin[i]) * ((output_vector[i] - LO) / (HI - LO)) + this.normalizeMin[i];
      unscaledVector[i] = temp;
    }

    return unscaledVector;
  },
  Run:function (dataX, testDataX, epoch) {
    if (!epoch)epoch = 1000;
    dataX = makeInt(dataX);
    testDataX = makeInt(testDataX);

    /* --- Threshhold - thresh (value of target mse, training stops once it is achieved) */
    Thresh = 0.000001;
    numEpoch = epoch;
    MSE = 0.0;
    this.NumPattern = dataX.length;
    this.NumInput = dataX[0].length;

    this.normalizeMax = new Array(this.NumPattern);
    this.normalizeMin = new Array(this.NumPattern);

    /* --- Pre-process data: Scale input and test values */
    consolelog('Call Scale Data');
    data = this.scale(clone(dataX));


    /* --- Start training: looping through epochs and exit when MSE error < Threshold */
    consolelog("\nNow training the network...." + numEpoch);
    //consolelog(dataX);
    for (var e = 0; e < numEpoch; e++) {
      /* -- Backpropagate */

      for (var d = 0; d < data.length; d++) {
        this.bpgt(data[d], data[d][data[d].length - 1]);
      }

      MSE = this.mse(data[e % data.length][data[e % data.length].length - 1]);
      //consolelog("MSE : " + MSE);
      if (e % 100 == 0)
        consolelog("Epoch : " + (e + 1) + " : " + MSE);
      if (e == 0) {
        consolelog("\nFirst epoch Mean Square Error: " + MSE);
      }

      if (MSE < Thresh) {
        consolelog("\nNetwork Trained. Threshold value achieved in " + (e + 1) + " iterations.");
        consolelog("\nMSE:  " + MSE);
        break;
      }
    }

    consolelog("\nLast epoch Mean Square Error: " + MSE);

    consolelog("\nNow using the trained network to make predictions on test data....\n");
    //consolelog(testDataX);
    var ttestDataX = new clone(testDataX);

    //this.normalizeMin = new Array();
    //this.normalizeMax = new Array();
    ttestDataX = this.scale(ttestDataX);
    for (var i = 0; i < ttestDataX.length; i++) {
      if (!(ttestDataX[i] instanceof Array))continue;
      this.ffwd(ttestDataX[i]);
      this.vectorOutput.push(this.Out(0));
    }

    //return;
    out = this.unscaleOutput(this.vectorOutput);


    var pl = '';
    for (col = 1; col < this.NumInput; col++) {
      pl += ("Inputcol\t");
    }
    consolelog(pl + "Predicted \n");

    for (i = 0; i < this.NumPattern; i++) {
      if (!(testDataX[i] instanceof Array))continue;
      var o = '';
      consolelog(testDataX[i]);
      for (j = 0; j < this.NumInput - 1; j++) {
        o += ("  " + testDataX[i][j] + ",");
      }
      consolelog(o + "::" + abs(out[i]) + "\n");
    }

  }

};

function Start() {
  /* --- Sample use */
// Mutliplication data: 1 x 1 = 1, 1 x 2 = 2,.. etc
  var dataSetx = [
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 3],
    [1, 4, 4],
    [1, 5, 5],
    [2, 1, 2],
    [2, 2, 4],
    [2, 3, 6],
    [2, 4, 8],
    [2, 5, 10],
    [3, 1, 3],
    [3, 2, 6],
    [3, 3, 9],
    [3, 4, 12],
    [3, 5, 15],
    [4, 1, 4],
    [4, 2, 8],
    [4, 3, 12],
    [4, 4, 16],
    [4, 5, 20],
    [5, 1, 5],
    [5, 2, 10],
    [5, 3, 15],
    [5, 4, 20],
    [5, 5, 25]
  ];
  consolelog(dataSetx);
// 1 x 1 =?
  var testDataSet = [
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [2, 1],
    [2, 2],
    [2, 3],
    [2, 4],
    [2, 5],
    [3, 1],
    [3, 2],
    [3, 3],
    [3, 4],
    [3, 5],
    [4, 1],
    [4, 2],
    [4, 3],
    [4, 4],
    [4, 5],
    [5, 1],
    [5, 2],
    [5, 3],
    [5, 4],
    [5, 5]
  ];

  layersSize = [2, 2, 1];
  numLayers = layersSize.length;

// Learing rate - beta
// momentum - alpha
  beta = 0.3;
  alpha = 0.1;

  minX = 1;
  maxX = 25;

// Creating the net
//bp=new BackPropagationScale(numLayers,layersSize,beta,alpha,minX,maxX);
  BackPropagationScale.initialize(numLayers, layersSize, beta, alpha, minX, maxX);
  BackPropagationScale.Run(dataSetx, testDataSet);
}

//Start();