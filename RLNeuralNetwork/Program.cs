using System;

namespace RLNeuralNetwork
{
    abstract class NetworkLayer
    {
        internal double[,] weights;
        internal double[] lastFeedForward;
        internal double[,] losses;
        internal NetworkLayer previousLayer;
        internal double[] previousInputs;
        public NetworkLayer(int inputs, int nodes)
        {
            Random r = new Random();
            weights = new double[inputs, nodes];
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = (double)r.NextDouble() * 5;
                }
            }
        }
        public NetworkLayer(double[,] weights)
        {
            previousLayer = null;
            this.weights = weights;
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
                    Console.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                nodes[i] = activationFunction(nodes[i]);
                //Console.WriteLine(nodes[i]);
            }
            lastFeedForward = nodes;
            return nodes;
        }


        internal abstract double activationFunction(double x);

        internal abstract double activationDerivative(double x);
        internal virtual void backPropagate()
        {
            double learningRate = 0.01;
            if (previousLayer != null)
            {
                previousLayer.losses = new double[previousLayer.weights.GetLength(0), previousLayer.weights.GetLength(1)];
                for (int i = 0; i < previousLayer.weights.GetLength(0); i++)
                {
                    for (int j = 0; j < previousLayer.weights.GetLength(1); j++)
                    {
                        previousLayer.losses[i, j] = weights[i, j] * losses[i, j] * activationDerivative(previousLayer.lastFeedForward[i]);
                    }
                }
            }
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = weights[i, j] - (learningRate * previousInputs[i] * losses[i, j]);
                }
            }
        }
    }

    class OutputLayer : NetworkLayer
    {
        public OutputLayer(int inputs, int nodes) : base(inputs, nodes) { }
        public OutputLayer(double[,] weights) : base(weights) { }

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
        public HiddenLayer(int inputs, int nodes) : base(inputs, nodes) {   }

        public HiddenLayer(double[,] weights) : base(weights) {   }

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

        internal override void backPropagate()
        {
            throw new NotImplementedException();
        }
    }

    class NeuralNetwork
    {
        HiddenLayer[] hiddenLayers;
        NetworkLayer outputLayer;

        public NeuralNetwork(HiddenLayer[] hiddenLayers, NetworkLayer outputLayer)
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
                        Console.WriteLine($"adding {equalInputs[k]} * {layer.weights[k + extraInputWeightIndex, j]} to {layerNodes[j]}");
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

        public void backPropagate(double[] inputs, double[] actualValues)
        {
            double[] ff = feedForward(inputs);
            double[,] losses = new double[ff.Length, 1];
            for (int i = 0; i < losses.Length; i++)
            {
                losses[i, 0] = ff[i] - actualValues[i];
            }
            outputLayer.losses = losses;
            outputLayer.backPropagate();
            for (int i = hiddenLayers.Length - 1; i >= 0; i--)
            {
                hiddenLayers[i].backPropagate();
            }
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

            double[] inputs;
            double value;
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
                        inputs.Add(double.Parse(inputsStrings[k]));
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

        void read()
        {
            Console.WriteLine("Enter filepath:");
            string path = Console.ReadLine();
            InputData[] data = ReadOverwatchResults(path);
            Console.WriteLine("bean");
        }

        static void Main(string[] args)
        {
            List<HiddenLayer> layers = new List<HiddenLayer>();
            //layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.6, -0.4 }, { 0.8, -0.7, -0.5 } }));
            layers.Add(new HiddenLayer(new double[,] { { 0.3, 0.8 }, { 0.6, -0.7 }, { -0.4, -0.5 } }));

            NeuralNetwork NN = new NeuralNetwork(layers.ToArray(), new OutputLayer(new double[,] { { -0.3 }, { 0.2 } }));
            Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            Console.WriteLine(NN.optimisedFeedForward(new double[] { 0.36, -1 }, new double[,] { { 0.4 } })[0,0]);
            //NN.backPropagate(new double[] { 0.36, -1, 0.4 }, new double[] { 5 });
            double e = 2;
        }
    }
}