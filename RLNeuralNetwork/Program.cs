using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace RLNeuralNetwork
{
    abstract class NetworkLayer
    {
        internal double[,] weights;
        internal double[] lastFeedForward;
        internal double[] losses;
        internal NetworkLayer previousLayer;
        internal double[] previousInputs;
        internal bool areLossesValid = false;
        internal double baseLearningRate;
        public NetworkLayer(int inputs, int nodes, double baseLearningRate)
        {
            Random r = new Random();
            weights = new double[inputs, nodes];
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = (double)r.NextDouble() * 0.00001;
                }
            }
            this.baseLearningRate = baseLearningRate;
        }
        public NetworkLayer(double[,] weights, double baseLearningRate)
        {
            previousLayer = null;
            this.weights = weights;
            this.baseLearningRate = baseLearningRate;
        }

        public double[] feedforward(double[] previousLayer)
        {
            //TODO: throw exception for previousLayer/weight size mismatch
            double[] nodes = new double[weights.GetLength(1)];
            for (int i = 0; i < nodes.Length; i++)
            {
                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    nodes[i] += previousLayer[j] * weights[j, i];
                    //.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                nodes[i] = activationFunction(nodes[i]);
                //Console.WriteLine(nodes[i]);
            }
            previousInputs = previousLayer;
            lastFeedForward = nodes;
            return nodes;
        }


        internal abstract double activationFunction(double x);

        internal abstract double activationDerivative(double x);
        internal virtual void backPropagate(double learningRateMulti = 1)
        {
            double learningRate = learningRateMulti * baseLearningRate;
            if (!areLossesValid)
            {
                throw new InvalidOperationException("Please begin backpropagation using backpropagate(losses) instead.");
            }
            areLossesValid = false;
            if (previousLayer != null)
            {
                previousLayer.losses = new double[weights.GetLength(0)];
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        previousLayer.losses[i] += weights[i, j] * losses[j];
                    }
                    previousLayer.losses[i] *= activationDerivative(previousLayer.lastFeedForward[i]);
                }
                previousLayer.areLossesValid = true;
            }
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = weights[i, j] - (learningRate * previousInputs[i]  * losses[j]);
                }
            }
            if (previousLayer != null)
            {
                previousLayer.backPropagate(learningRateMulti: learningRateMulti);
            }
        }

        internal virtual void backPropagate(double[] losses, double learningRateMulti)
        {
            this.losses = losses;
            areLossesValid = true;
            backPropagate(learningRateMulti: learningRateMulti);
        }
    }

    class OutputLayer : NetworkLayer
    {
        public OutputLayer(int inputs, int nodes, double baseLearningRate) : base(inputs, nodes, baseLearningRate) { }
        public OutputLayer(double[,] weights, double baseLearningRate) : base(weights, baseLearningRate) { }

        internal override double activationFunction(double x)
        {
            return x;
        }

        internal override double activationDerivative(double x)
        {
            return 1;
        }
    }

    class HiddenLayer : NetworkLayer
    {
        public HiddenLayer(int inputs, int nodes, double baseLearningRate) : base(inputs, nodes, baseLearningRate) {   }

        public HiddenLayer(double[,] weights, double baseLearningRate) : base(weights, baseLearningRate) {   }

        internal override double activationFunction(double x)
        {
            double e2x = Math.Pow(Math.E, 2 * x);
            return (e2x - 1) / (e2x + 1);
        }

        internal override double activationDerivative(double x)
        {
            double y = activationFunction(x);
            return 1 - (y * y);
        }
    }

    class NeuralNetwork
    {
        HiddenLayer[] hiddenLayers;
        OutputLayer outputLayer;

        public NeuralNetwork(HiddenLayer[] hiddenLayers, OutputLayer outputLayer)
        {
            this.hiddenLayers = hiddenLayers;
            for (int i = 1; i < hiddenLayers.Length; i++)
            {
                hiddenLayers[i].previousLayer = hiddenLayers[i - 1];
            }
            this.outputLayer = outputLayer;
            if (hiddenLayers.Length > 0)
            {
                outputLayer.previousLayer = hiddenLayers[hiddenLayers.Length - 1];
            }
        }

        public double[] feedForward(double[] inputs)
        {
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                inputs = hiddenLayers[i].feedforward(inputs);
            }
            return outputLayer.feedforward(inputs);
        }

        /// <summary>
        /// An optimised feed-forward intended for multiple input sets where some of their inputs are equal. 
        /// Precomputes the first hidden layer using the common inputs to re-use for each unique input set.
        /// This implementation requires all of the common inputs to come before the unique inputs (i.e., the inputs have the same order as they would during regular feed-forward).
        /// </summary>
        /// <param name="equalInputs">The inputs that each input set has in common.</param>
        /// <param name="uniqueInputs">The set of inputs that are unique to each input set.</param>
        /// <returns></returns>
        public double[,] optimisedFeedForward(double[] equalInputs, double[,] uniqueInputs)
        {
            if (hiddenLayers.Length == 0)
            {
                throw new Exception("Optimised feed-forward requires a hidden layer.");
            }

            //precompute first hidden layer
            HiddenLayer layer = hiddenLayers[0];
            double[] layerNodes = new double[layer.weights.GetLength(1)];
            for (int i = 0; i < layer.weights.GetLength(1); i++)
            {
                for (int j = 0; j < equalInputs.Length; j++)
                {
                    layerNodes[i] += equalInputs[j] * layer.weights[j, i];
                    //Console.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                //Console.WriteLine(nodes[i]);
            }

            double[] layer1NodesBackup = layerNodes;
            int extraInputWeightIndex = equalInputs.Length;
            double[,] results = new double[uniqueInputs.GetLength(0), outputLayer.weights.GetLength(1)];

            //foreach unique input set
            for (int i = 0; i < uniqueInputs.GetLength(0); i++)
            {
                layerNodes = layer1NodesBackup;
                //finish computing first layer
                for (int j = 0; j < layerNodes.Length; j++)
                {
                    for (int k = 0; k < uniqueInputs.GetLength(1); k++)
                    {
                        layerNodes[j] += uniqueInputs[i, k] * layer.weights[k + extraInputWeightIndex, j];
                        //Console.WriteLine($"adding {equalInputs[k]} * {layer.weights[k + extraInputWeightIndex, j]} to {layerNodes[j]}");
                    }
                    layerNodes[j] = layer.activationFunction(layerNodes[j]);
                }

                //compute other layers as normal
                for (int j = 1; j < hiddenLayers.Length; j++)
                {
                    layerNodes = feedForward(layerNodes);
                }
                layerNodes = outputLayer.feedforward(layerNodes);

                for (int j = 0; j < results.GetLength(1); j++)
                {
                    results[i, j] = layerNodes[j];
                }
            }

            return results;
        }

        public void backPropagate(double[] inputs, double[] actualValues, double learningRateMulti, bool printLoss = false)
        {
            double[] ff = feedForward(inputs);
            double[] losses = new double[ff.Length];
            for (int i = 0; i < losses.Length; i++)
            {
                losses[i] = ff[i] - actualValues[i];
            }
            if (printLoss)
            {
                Console.WriteLine(losses[0]);
            }
            outputLayer.backPropagate(losses, learningRateMulti);
        }
    }

    internal class Program
    {

        internal struct InputData
        {
            public InputData(double[] inputs, double value)
            {
                this.inputs = inputs;
                this.value = value;
            }

            public double[] inputs;
            public double value;
        }

        static InputData[] ReadOverwatchResults(string filePath)
        {
            StreamReader sr = new StreamReader(filePath);
            string arrayString = sr.ReadToEnd();
            arrayString = arrayString.Replace("\t", "").Replace(";","");
            string[] arraysByMatch = arrayString.Split("Global.inputResults = ");
            List<InputData> inputDatas = new List<InputData>();
            for (int i = 1; i < arraysByMatch.Length; i++)
            {
                arraysByMatch[i] = arraysByMatch[i].Substring(6);
                if (i + 1 == arraysByMatch.Length)
                {
                    arraysByMatch[i] = arraysByMatch[i].Remove(arraysByMatch[i].Length - 1);
                } 
                else
                {
                    arraysByMatch[i] = arraysByMatch[i].Remove(arraysByMatch[i].Length - 3);
                }

                arraysByMatch[i] = arraysByMatch[i].Replace("\r\n", "");
                arraysByMatch[i] = arraysByMatch[i].Replace(" ", "");

                string[] dataStrings = arraysByMatch[i].Split("Array(");

                for (int j = 1; j < dataStrings.Length; j++)
                {
                    string inputsString = dataStrings[j].Split("),")[0];
                    string[] inputsStrings = inputsString.Split(',');
                    string valueString = dataStrings[j].Split("),")[1];
                    valueString = valueString.Replace(",", "");

                    List<double> inputs = new List<double>();
                    for (int k = 0; k < inputsStrings.Length; k++)
                    {
                        double input = double.Parse(inputsStrings[k]);
                        if (k >= 49 && k <= 51)
                        {
                            if (input < 0)
                            {
                                input = -0.033;
                            }
                        }
                        inputs.Add(input);
                    }
                    double value = double.Parse(valueString);
                    inputDatas.Add(new InputData(inputs.ToArray(), value));
                }
                
            }
            return inputDatas.ToArray();
        }

        void ToOverwatchArray()
        {

        }

        static InputData[] read()
        {
            Console.WriteLine("Enter filepath:");
            string path = Console.ReadLine();
            InputData[] data = ReadOverwatchResults(path);
            return data;
        }
        
        void test()
        {
            List<HiddenLayer> layers = new List<HiddenLayer>();
            //layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.6, -0.4 }, { 0.8, -0.7, -0.5 } }));
            layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.8 }, { 0.6, -0.7 }, { -0.4, -0.5 } }, 0.1));

            NeuralNetwork NN = new NeuralNetwork(layers.ToArray(), new OutputLayer(new double[,] { { -0.3 }, { 0.2 } }, 0.01));
            Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            NN.backPropagate(new double[] { 0.36, -1, 0.4 }, new double[] { 5 }, 1);
            Console.WriteLine(NN.optimisedFeedForward(new double[] { 0.36, -1 }, new double[,] { { 0.4 } })[0, 0]);

            for (int i = 0; i < 1000; i++)
            {
                //NN.backPropagate(new double[] { 0.36, -1, 0.4 }, new double[] { 5 });
                //Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            }
        }

        void test2()
        {
            NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(1000, 800, 0.1) }, new OutputLayer(800, 1, 0.01));
            double[] test = randomDoubleArray(1000);
            double[] test2 = randomDoubleArray(1000);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            double result = NN.feedForward(test)[0];
            sw.Stop();
            Console.WriteLine(result);
            for (int i = 0; i < 100; i++)
            {
                NN.backPropagate(test, new double[] { 500 }, 1);
                NN.backPropagate(test2, new double[] { -700 }, 1);
            }
            result = NN.feedForward(test)[0];
            Console.WriteLine(result);
            result = NN.feedForward(test2)[0];
            Console.WriteLine(result);
            Console.WriteLine(sw.ElapsedTicks);
        }

        static double[] randomDoubleArray(int length)
        {
            Random random = new Random();
            double[] array = new double[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = random.NextDouble() * 0.0005;
            }
            return array;
        }

        static void Main(string[] args)
        {
            double metaLoss = 0;
            double metaLossCount = 0;
            double metaLoss2 = 0;
            double metaLossCount2 = 0;
            InputData[] data = read();
            for (int i = 0; i < 1; i++)
            {
                NeuralNetwork NN = new NeuralNetwork(new HiddenLayer[] { new HiddenLayer(55, 55, 0.01), new HiddenLayer(55, 55, 0.01) }, new OutputLayer(55, 1, 0.001));
                for (int j = 0; j < 100; j++)
                {
                    foreach (InputData x in data)
                    {
                        NN.backPropagate(x.inputs, new double[] { x.value }, 1 / ((i*5)+1), printLoss: false);
                    }
                }
                double loss = 0;
                int lossCount = 0;
                double loss2 = 0;
                int lossCount2 = 0;
                foreach (InputData x in data)
                {
                    if (x.inputs[49] == 0.1 || x.inputs[50] == 0.1 || x.inputs[51] == 0.1)
                    {
                        loss2 += Math.Abs(x.value - NN.feedForward(x.inputs)[0]);
                        lossCount2++;
                    } else
                    {
                        loss += Math.Abs(x.value - NN.feedForward(x.inputs)[0]);
                        lossCount++;
                    }
                }
                metaLoss += loss / lossCount;
                metaLossCount++;
                metaLoss2 += loss2 / lossCount2;
                metaLossCount2++;
            }
            Console.WriteLine(metaLoss / metaLossCount);
            Console.WriteLine(metaLoss2 / metaLossCount2);
            Console.WriteLine((metaLoss + metaLoss2) / (metaLossCount + metaLossCount2));
            Console.WriteLine("ee");
        }
    }
}