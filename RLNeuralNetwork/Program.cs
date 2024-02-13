using System;

namespace RLNeuralNetwork // Note: actual namespace depends on the project name.
{
    class NetworkLayer
    {
        internal double[,] weights;
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
            this.weights = weights;
        }

        public double[] feedforward(double[] previousLayer)
        {
            double[] nodes = new double[weights.GetLength(1)];
            for (int i = 0; i < nodes.Length; i++)
            {
                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    nodes[i] += previousLayer[j] * weights[j, i];
                    Console.WriteLine($"adding {previousLayer[j]} * {weights[j, i]} to {nodes[i]}");
                }
                nodes[i] = activationFunction(nodes[i]);
                Console.WriteLine(nodes[i]);
            }
            return nodes;
        }


        internal virtual double activationFunction(double x)
        {
            return x;
        }

        internal virtual double activationDerivative(double x)
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
    }
    class NeuralNetwork
    {
        HiddenLayer[] hiddenLayers;
        NetworkLayer outputLayer;

        public NeuralNetwork(HiddenLayer[] hiddenLayers, NetworkLayer outputLayer)
        {
            this.hiddenLayers = hiddenLayers;
            this.outputLayer = outputLayer;
        }

        public double[] feedForward(double[] inputs)
        {
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                inputs = hiddenLayers[i].feedforward(inputs);
            }
            return outputLayer.feedforward(inputs);
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

        static InputData[]? ReadOverwatchResults(string filePath)
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

            NeuralNetwork NN = new NeuralNetwork(layers.ToArray(), new NetworkLayer(new double[,] { { -0.3 }, { 0.2 } }));
            Console.WriteLine(NN.feedForward(new double[] { 0.36, -1, 0.4 })[0]);
            double e = 2;
        }
    }
}