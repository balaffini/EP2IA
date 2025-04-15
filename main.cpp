#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <cmath>
#include <array>

#define max_epochs 1050
#define error_threshold 0.01

#define with_momentum false
#define with_early_stopping true

#define DATA_COLUMNS 120 // Dimensão dos dados
#define DATA_ROWS 1326 // Quantidade total de dados
#define LABEL_SIZE 26 // Tamanho do vetor de saída
#define SPLIT_INDEX 1066 // Quantidade de dados de treino vs. teste (teste = DATA_ROWS - SPLIT_INDEX)
#define HIDDEN_LAYER_SIZE 106 // Tamanho da camada escondida
#define TRAIN_DATA_SIZE 1066 // Quantidade de dados de treino
#define TEST_DATA_SIZE 260 // Quantidade de dados de teste (TEST_DATA_SIZE = DATA_ROWS - TRAIN_DATA_SIZE)
#define LEARNING_RATE 0.075 // Taxa de aprendizado
#define DECAY_RATE (1.0 / max_epochs) // Taxa de decaimento da taxa de aprendizado
#define K_FOLDS 8 // Quantidade de folds para cross-validation
#define SAVE_ERROR false // Salvar os erros em um arquivo
#define FILENAME_DATA "X.txt" // Nome do arquivo de dados
#define FILENAME_LABELS "Y_letra.txt" // Nome do arquivo de rótulos
#define FILENAME_ERRORS ("total_errors_" + std::to_string(HIDDEN_LAYER_SIZE) + "_" + std::to_string(LEARNING_RATE) + ".csv")
#define FILENAME_CONFUSION_MATRIX ("confusion_matrix_" + std::to_string(HIDDEN_LAYER_SIZE) + "_" + std::to_string(LEARNING_RATE) + ".txt")

#define FILENAME_WEIGHTS ("weights_" + std::to_string(HIDDEN_LAYER_SIZE) + "_" + std::to_string(LEARNING_RATE) + ".txt")

namespace {
    /**
 * @brief Generates an array of random numbers.
 *
 * This function generates an array of random numbers using the Mersenne Twister algorithm.
 * The size of the array is determined by the template parameter `N`.
 * The generated numbers are uniformly distributed between `-(1.0 / sqrt(N))` and `(1.0 / sqrt(N))`.
 *
 * @tparam N The size of the array to be generated.
 *
 * @return An array of `N` random numbers.
 */
    template<size_t N>
    std::array<double, N> randnumb() {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> distribution(-(1.0 / (std::sqrt(N))), (1.0 / (std::sqrt(N))));
        std::array<double, N> random_values{};

        for (int i = 0; i < N; ++i) {
            random_values[i] = distribution(gen);
        }

        return random_values;
    }

    template<int NumDataColumns, int NumDataRows, int LabelSize, int SplitIndex>
    std::pair<std::pair<std::array<std::array<double, NumDataColumns>, SplitIndex>, std::array<std::array<double, NumDataColumns>,
            NumDataRows -
            SplitIndex>>, std::pair<std::array<std::array<double, LabelSize>, SplitIndex>, std::array<std::array<double, LabelSize>,
            NumDataRows - SplitIndex>>>
    splitData(const std::array<std::array<double, NumDataColumns>, NumDataRows> &data,
              const std::array<std::array<double, LabelSize>, NumDataRows> &labels) {
        std::array<std::array<double, NumDataColumns>, SplitIndex> trainData{};
        std::array<std::array<double, NumDataColumns>, NumDataRows - SplitIndex> testData{};
        std::array<std::array<double, LabelSize>, SplitIndex> trainLabels{};
        std::array<std::array<double, LabelSize>, NumDataRows - SplitIndex> testLabels{};

        std::copy(data.begin(), data.begin() + SplitIndex, trainData.begin());
        std::copy(data.begin() + SplitIndex, data.end(), testData.begin());
        std::copy(labels.begin(), labels.begin() + SplitIndex, trainLabels.begin());
        std::copy(labels.begin() + SplitIndex, labels.end(), testLabels.begin());

        return {{trainData,   testData},
                {trainLabels, testLabels}};
    }


    template<int Dimension, int NumClasses, int HiddenLayerSize>
    class MultiLayerPerceptron {
    private:
        std::array<std::array<double, Dimension>, HiddenLayerSize> weights_input_to_hidden; // Input -> Hidden
        std::array<double, HiddenLayerSize> bias_weight_input_to_hidden; // Input -> Hidden
        std::array<std::array<double, HiddenLayerSize>, NumClasses> weights_hidden_to_output; // Hidden -> Output
        std::array<double, NumClasses> bias_weight_hidden_to_output; // Hidden -> Output

        std::array<double, HiddenLayerSize> hidden_layer_output; // y
        std::array<double, NumClasses> output_layer_product; // z_in
        std::array<double, HiddenLayerSize> hidden_layer_product; // y_in

        std::vector<std::vector<double>> training_data_buffer; // Buffer to store intermediate values; // Buffer to store intermediate values

        double learning_rate;
        const double decay_rate;

        [[nodiscard]] static double act_func(const double &net) { // função sigmoid
            return 1.0 / (1.0 + std::exp(-net));
        }

        template<size_t N>
        [[nodiscard]] static double
        dot_product(const std::array<double, N> &data, const std::array<double, N> &weight, const double &bias) {
            return std::inner_product(data.begin(), data.end(), weight.begin(), bias);
        }

        template<size_t N>
        [[nodiscard]] static double
        dot_product(const std::array<int, N> &data, const std::array<double, N> &weight, const double &bias) {
            return std::inner_product(data.begin(), data.end(), weight.begin(), bias);
        }

        [[nodiscard]] static double delt_act_func(const double &net) {
            double f_net = 1.0 / (1.0 + std::exp(-net));
            return f_net * (1.0 - f_net);
        }

        /**
 * @brief Trains the neural network using the momentum method.
 *
 * This function trains the neural network using the momentum method, which is a method that helps accelerate
 * SGD in the relevant direction and dampens oscillations. It does this by adding a fraction of the update vector
 * of the past time step to the current update vector.
 *
 * @tparam DataSize The number of data points in the dataset.
 * @param dataset The dataset used to train the network. It is a 2D array where each row is a data point and each column is a feature.
 * @param target The expected output values for each data point in the dataset. It is a 2D array where each row corresponds to a data point and each column is a class.
 * @return The total error of the network after training.
 */
        template<int DataSize>
        double internal_train_momentum(const std::array<std::array<double, Dimension>, DataSize> &dataset,
                                       const std::array<std::array<double, NumClasses>, DataSize> &target) {
            // Initialize momentum-related variables
            std::array<std::array<double, Dimension>, HiddenLayerSize> delta_w_momentum_input_hidden{};
            std::array<double, HiddenLayerSize> delta_b_momentum_input_hidden{};
            std::array<std::array<double, HiddenLayerSize>, NumClasses> delta_w_momentum_hidden_output{};
            std::array<double, NumClasses> delta_b_momentum_hidden_output{};
            const double alpha = 0.9;
            const double beta = 0.9;

            double total_error = 0.0;
            for (int i = 0; i < DataSize; ++i) {
                // Forward pass
                const std::array<double, NumClasses> outputs = predict(dataset[i]);

                // Calculate the errors for the output layer
                std::array<double, NumClasses> errors_output{};
                for (int k = 0; k < NumClasses; ++k) {
                    errors_output[k] = target[i][k] - outputs[k];
                    total_error += errors_output[k] * errors_output[k];
                    errors_output[k] *= delt_act_func(output_layer_product[k]);
                }

                // Backpropagation for hidden-to-output weights (∆wjk)
                std::array<std::array<double, HiddenLayerSize>, NumClasses> delta_hidden_from_output{};
                for (int k = 0; k < NumClasses; ++k) {
                    for (int j = 0; j < HiddenLayerSize; ++j) {
                        delta_hidden_from_output[k][j] = learning_rate * errors_output[k] * hidden_layer_output[j];
                        delta_w_momentum_hidden_output[k][j] = alpha * delta_w_momentum_hidden_output[k][j] +
                                                               beta * delta_hidden_from_output[k][j];
                        weights_hidden_to_output[k][j] += delta_w_momentum_hidden_output[k][j];
                    }
                }

                // Backpropagation for output bias weights (∆w0k)
                std::array<double, NumClasses> delta_output_bias{};
                for (int k = 0; k < NumClasses; ++k) {
                    delta_output_bias[k] = learning_rate * errors_output[k];
                    delta_b_momentum_hidden_output[k] =
                            alpha * delta_b_momentum_hidden_output[k] + beta * delta_output_bias[k];
                    bias_weight_hidden_to_output[k] += delta_b_momentum_hidden_output[k];
                }

                // Calculate errors for the hidden layer
                std::array<double, HiddenLayerSize> errors_hidden{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    double delta_in_j = 0.0;
                    for (int k = 0; k < NumClasses; ++k) {
                        delta_in_j += weights_hidden_to_output[k][j] * errors_output[k];
                    }
                    errors_hidden[j] = delta_in_j * delt_act_func(hidden_layer_product[j]);
                }

                // Backpropagation for input-to-hidden weights (∆vij)
                std::array<std::array<double, Dimension>, HiddenLayerSize> delta_input_from_hidden{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    for (int l = 0; l < Dimension; ++l) {
                        delta_input_from_hidden[j][l] = learning_rate * errors_hidden[j] * dataset[i][l];
                        delta_w_momentum_input_hidden[j][l] = alpha * delta_w_momentum_input_hidden[j][l] +
                                                              beta * delta_input_from_hidden[j][l];
                        weights_input_to_hidden[j][l] += delta_w_momentum_input_hidden[j][l];
                    }
                }

                // Backpropagation for hidden bias weights (∆v0j)
                std::array<double, HiddenLayerSize> delta_hidden_bias{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    delta_hidden_bias[j] = learning_rate * errors_hidden[j];
                    delta_b_momentum_input_hidden[j] =
                            alpha * delta_b_momentum_input_hidden[j] + beta * delta_hidden_bias[j];
                    bias_weight_input_to_hidden[j] += delta_b_momentum_input_hidden[j];
                }
            }

            total_error = total_error / static_cast<double>(dataset.size());
            return total_error;
        }

        template<int DataSize>
        double internal_train(const std::array<std::array<double, Dimension>, DataSize> &dataset,
                              const std::array<std::array<double, NumClasses>, DataSize> &target) {
            double total_error = 0.0;
            for (int i = 0; i < DataSize; ++i) {
                // Forward pass
                const std::array<double, NumClasses> outputs = predict(dataset[i]);

                // Calculate the errors for the output layer
                std::array<double, NumClasses> errors_output{};
                for (int k = 0; k < NumClasses; ++k) {
                    errors_output[k] = target[i][k] - outputs[k];
                    total_error += errors_output[k] * errors_output[k];
                    errors_output[k] *= delt_act_func(output_layer_product[k]);
                }

                // Calculate delta_hidden_from_output (∆wjk)
                std::array<std::array<double, HiddenLayerSize>, NumClasses> delta_hidden_from_output{};
                for (int k = 0; k < NumClasses; ++k) {
                    for (int j = 0; j < HiddenLayerSize; ++j) {
                        delta_hidden_from_output[k][j] = learning_rate * errors_output[k] * hidden_layer_output[j];
                    }
                }

                // Calculate delta_output_bias (∆w0k)
                std::array<double, NumClasses> delta_output_bias{};
                for (int k = 0; k < NumClasses; ++k) {
                    delta_output_bias[k] = learning_rate * errors_output[k];
                }

                // Calculate errors for the hidden layer
                std::array<double, HiddenLayerSize> errors_hidden{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    double delta_in_j = 0.0;
                    for (int k = 0; k < NumClasses; ++k) {
                        delta_in_j += weights_hidden_to_output[k][j] * errors_output[k];
                    }
                    errors_hidden[j] = delta_in_j * delt_act_func(hidden_layer_product[j]);
                }

                // Calculate delta_input_from_hidden (∆vij)
                std::array<std::array<double, Dimension>, HiddenLayerSize> delta_input_from_hidden{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    for (int l = 0; l < Dimension; ++l) {
                        delta_input_from_hidden[j][l] = learning_rate * errors_hidden[j] * dataset[i][l];
                    }
                }

                // Calculate delta_hidden_bias (∆v0j)
                std::array<double, HiddenLayerSize> delta_hidden_bias{};
                for (int j = 0; j < HiddenLayerSize; ++j) {
                    delta_hidden_bias[j] = learning_rate * errors_hidden[j];
                }

                // Update the weights for all layers
                for (int k = 0; k < NumClasses; ++k) {
                    for (int j = 0; j < HiddenLayerSize; ++j) {
                        weights_hidden_to_output[k][j] += delta_hidden_from_output[k][j];
                    }
                    bias_weight_hidden_to_output[k] += delta_output_bias[k];
                }

                for (int j = 0; j < HiddenLayerSize; ++j) {
                    for (int l = 0; l < Dimension; ++l) {
                        weights_input_to_hidden[j][l] += delta_input_from_hidden[j][l];
                    }
                    bias_weight_input_to_hidden[j] += delta_hidden_bias[j];
                }
            }

            total_error = total_error / static_cast<double>(DataSize);
            return total_error;
        }

    public:
        MultiLayerPerceptron(double learning_rate, double decay_rate)
                : learning_rate(learning_rate), decay_rate(decay_rate) {
            // Initialize each inner array with a separate call to randnumb
            for (int i = 0; i < NumClasses; ++i) {
                weights_hidden_to_output[i] = randnumb<HiddenLayerSize>();
            }

            // Initialize each inner array with a separate call to randnumb
            for (int i = 0; i < HiddenLayerSize; ++i) {
                weights_input_to_hidden[i] = randnumb<Dimension>();
            }

            bias_weight_hidden_to_output = randnumb<NumClasses>();
            bias_weight_input_to_hidden = randnumb<HiddenLayerSize>();
        }

        template<int DataSize>
        double calculate_error(const std::array<std::array<double, Dimension>, DataSize> &dataset,
                               const std::array<std::array<double, NumClasses>, DataSize> &target) {
            double total_error = 0.0;
            for (int i = 0; i < DataSize; ++i) {
                // Forward pass
                const std::array<double, NumClasses> outputs = predict(dataset[i]);

                // Calculate the errors for the output layer
                std::array<double, NumClasses> errors_output{};
                for (int k = 0; k < NumClasses; ++k) {
                    errors_output[k] = target[i][k] - outputs[k];
                    total_error += errors_output[k] * errors_output[k];
                }
            }

            total_error = total_error / static_cast<double>(DataSize);
            return total_error;
        }

        template<int DataSize>
        std::pair<std::array<std::array<double, Dimension>, DataSize>, std::array<std::array<double, NumClasses>, DataSize>>
        shuffleData(std::array<std::array<double, Dimension>, DataSize> &dataset,
                    std::array<std::array<double, NumClasses>, DataSize> &target,
                    std::array<int, DataSize> &indices) {
            // Shuffle indices
            std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

            // Create temporary arrays for shuffled data
            std::array<std::array<double, Dimension>, DataSize> shuffled_dataset{};
            std::array<std::array<double, NumClasses>, DataSize> shuffled_target{};

            // Reorder data based on shuffled indices
            for (size_t i = 0; i < dataset.size(); ++i) {
                shuffled_dataset[i] = dataset[indices[i]];
                shuffled_target[i] = target[indices[i]];
            }

            return {shuffled_dataset, shuffled_target};
        }

        template<int DataSize, bool WithMomentum>
        void train(std::array<std::array<double, Dimension>, DataSize> &dataset,
                   std::array<std::array<double, NumClasses>, DataSize> &target) {
            const double initial_learning_rate = learning_rate;

            // Create and initialize index array
            std::array<int, DataSize> indices{};
            std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
            double total_error;

            for (int epoch = 0; epoch < max_epochs; ++epoch) {
                if constexpr (WithMomentum) {
                    total_error = internal_train_momentum<DataSize>(dataset, target);
                } else {
                    total_error = internal_train<DataSize>(dataset, target);
                }

                // Print the MSE for each epoch
                //std::cout << "Epoch: " << epoch << ", MSE: " << total_error << "\n";

                if (total_error < error_threshold) {
                    break;
                }

                // Update learning rate using decay
                learning_rate = initial_learning_rate / (1.0 + (decay_rate * epoch));

                // Shuffle the data
                auto [shuffled_dataset, shuffled_target] = shuffleData<DataSize>(dataset, target, indices);

                // Update dataset and target arrays
                dataset = shuffled_dataset;
                target = shuffled_target;


                if constexpr (SAVE_ERROR) {
                    std::vector<double> data_point;
                    data_point.insert(data_point.end(), total_error);
                    training_data_buffer.push_back(data_point);
                }
            }
        }

        template<int DataSize, int TrainingSize, bool WithMomentum>
        void train(std::array<std::array<double, Dimension>, DataSize> &dataset,
                   std::array<std::array<double, NumClasses>, DataSize> &target,
                   int patience) {
            const double initial_learning_rate = learning_rate;

            // Split the data into training and validation sets
            auto [data_letras_split, label_letras_split] = splitData<Dimension, DataSize, NumClasses, TrainingSize>(
                    dataset,
                    target);

            // Create and initialize index array
            std::array<int, TrainingSize> indices{};
            std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
            double total_error;

            double best_val_error = std::numeric_limits<double>::max();
            int no_improvement_epochs = 0;

            for (int epoch = 0; epoch < max_epochs; ++epoch) {
                if constexpr (WithMomentum) {
                    total_error = internal_train_momentum<TrainingSize>(data_letras_split.first,
                                                                        label_letras_split.first);
                } else {
                    total_error = internal_train<TrainingSize>(data_letras_split.first, label_letras_split.first);
                }

                // Early stopping check
                double val_error =
                        calculate_error<DataSize - TrainingSize>(data_letras_split.second, label_letras_split.second);
                if (val_error < best_val_error) {
                    best_val_error = val_error;
                    no_improvement_epochs = 0;
                } else {
                    no_improvement_epochs++;
                    if (no_improvement_epochs >= patience) {
                        std::cout << "Early stopping after epoch " << epoch << " with validation error " << val_error
                                  << std::endl;
                        break;
                    }
                }

                if (total_error < error_threshold) {
                    break;
                }

                std::cout << "Epoch: \t" << epoch << ",\tMSE: \t" << total_error << ",\tVMSE: \t" << val_error << "\n";

                // Update learning rate using decay
                learning_rate = initial_learning_rate / (1.0 + (decay_rate * epoch));

                // Shuffle the data
                auto [shuffled_dataset, shuffled_target] = shuffleData<TrainingSize>(data_letras_split.first,
                                                                                     label_letras_split.first, indices);

                // Update dataset and target arrays
                data_letras_split.first = shuffled_dataset;
                label_letras_split.first = shuffled_target;

                if constexpr (SAVE_ERROR) {
                    std::vector<double> data_point;
                    data_point.insert(data_point.end(), total_error);
                    data_point.insert(data_point.end(), val_error);
                    training_data_buffer.push_back(data_point);
                }
            }
        }

        std::array<double, NumClasses> predict(const std::array<double, Dimension> &data) {

            for (int i = 0; i < HiddenLayerSize; ++i) {
                hidden_layer_product[i] = dot_product(data, weights_input_to_hidden[i], bias_weight_input_to_hidden[i]);
                hidden_layer_output[i] = act_func(hidden_layer_product[i]);
            }

            std::array<double, NumClasses> output{};
            for (int i = 0; i < NumClasses; ++i) {
                output_layer_product[i] = dot_product(hidden_layer_output, weights_hidden_to_output[i],
                                                      bias_weight_hidden_to_output[i]);
                output[i] = act_func(output_layer_product[i]);
            }

            return output;
        }

        void save_weights(const std::string &filename) {
            std::ofstream file(filename);

            // Save weights from input to hidden layer
            for (const auto &row: weights_input_to_hidden) {
                for (const auto &value: row) {
                    file << value << ' ';
                }
                file << '\n';
            }

            // Save bias weights from input to hidden layer
            for (const auto &value: bias_weight_input_to_hidden) {
                file << value << ' ';
            }
            file << '\n';

            // Save weights from hidden to output layer
            for (const auto &row: weights_hidden_to_output) {
                for (const auto &value: row) {
                    file << value << ' ';
                }
                file << '\n';
            }

            // Save bias weights from hidden to output layer
            for (const auto &value: bias_weight_hidden_to_output) {
                file << value << ' ';
            }
            file << '\n';

            file.close();
        }

        void load_weights(const std::string &filename) {
            std::ifstream file(filename);

            // Load weights from input to hidden layer
            for (auto &row: weights_input_to_hidden) {
                for (auto &value: row) {
                    file >> value;
                }
            }

            // Load bias weights from input to hidden layer
            for (auto &value: bias_weight_input_to_hidden) {
                file >> value;
            }

            // Load weights from hidden to output layer
            for (auto &row: weights_hidden_to_output) {
                for (auto &value: row) {
                    file >> value;
                }
            }

            // Load bias weights from hidden to output layer
            for (auto &value: bias_weight_hidden_to_output) {
                file >> value;
            }

            file.close();
        }


        void save_training_data(const std::string &filename) {
            std::ofstream outfile(filename);
            for (const auto &data_point: training_data_buffer) {
                std::copy(data_point.begin(), data_point.end(), std::ostream_iterator<double>(outfile, ", "));
                outfile << "\n";
            }
        }


        // Função para resetar os pesos da rede
        void reset_weights() {
            for (int i = 0; i < NumClasses; ++i) {
                weights_hidden_to_output[i] = randnumb<HiddenLayerSize>();
            }

            for (int i = 0; i < HiddenLayerSize; ++i) {
                weights_input_to_hidden[i] = randnumb<Dimension>();
            }

            bias_weight_hidden_to_output = randnumb<NumClasses>();
            bias_weight_input_to_hidden = randnumb<HiddenLayerSize>();

            // Reset the learning rate
            learning_rate = LEARNING_RATE;
        }
    };


/**
 * @brief Reads data from a file and returns it as a pair of 2D arrays.
 *
 * This function reads data from a file where each line represents a row of data and each value is separated by a comma.
 * The data is stored in a 2D array where each row corresponds to a line in the file and each column corresponds to a value in that line.
 * The function also handles the Byte Order Mark (BOM) if it is present in the file.
 *
 * @tparam NumDataColumns The number of columns in the data array.
 * @tparam NumDataRows The number of rows in the data array.
 * @tparam LabelSize The number of columns in the labels array.
 *
 * @param filename The name of the file to read the data from.
 *
 * @return A pair of 2D arrays. The first array contains the data and the second array contains the labels.
 */
    template<int NumDataColumns, int NumDataRows, int LabelSize>
    std::pair<std::array<std::array<double, NumDataColumns>, NumDataRows>, std::array<std::array<double, LabelSize>, NumDataRows>>
    readData(const std::string &filename) {
        std::ifstream file(filename);
        std::string line;
        std::array<std::array<double, NumDataColumns>, NumDataRows> data{};
        std::array<std::array<double, LabelSize>, NumDataRows> labels{};

        int row_count = 0;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string value;

            // Read data values
            for (int i = 0; i < NumDataColumns; ++i) {
                std::getline(ss, value, ',');
                // Check for and remove BOM
                if (!value.empty() && value[0] == '\xEF' && value[1] == '\xBB' && value[2] == '\xBF') {
                    value = value.substr(3);
                }
                data[row_count][i] = std::stod(value);
            }

            // Read label values
            for (int j = 0; j < LabelSize; ++j) {
                std::getline(ss, value, ',');
                // Check for and remove BOM
                if (!value.empty() && value[0] == '\xEF' && value[1] == '\xBB' && value[2] == '\xBF') {
                    value = value.substr(3);
                }
                labels[row_count][j] = std::stod(value);
            }

            row_count++;
        }

        return {data, labels};
    }

/**
 * @brief Reads data from a file and returns it as a 2D array.
 *
 * This function reads data from a file where each line represents a row of data and each value is separated by a comma.
 * The data is stored in a 2D array where each row corresponds to a line in the file and each column corresponds to a value in that line.
 * The function also handles the Byte Order Mark (BOM) if it is present in the file.
 *
 * @tparam NumDataColumns The number of columns in the data array.
 * @tparam NumDataRows The number of rows in the data array.
 *
 * @param filename The name of the file to read the data from.
 *
 * @return A 2D array containing the data read from the file.
 */
    template<int NumDataColumns, int NumDataRows>
    std::array<std::array<double, NumDataColumns>, NumDataRows> readData(const std::string &filename) {
        std::ifstream file(filename);
        std::string line;
        std::array<std::array<double, NumDataColumns>, NumDataRows> data{};

        int row_count = 0;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string value;

            for (int i = 0; i < NumDataColumns; ++i) {
                std::getline(ss, value, ',');
                // Check for and remove BOM
                if (!value.empty() && value[0] == '\xEF' && value[1] == '\xBB' && value[2] == '\xBF') {
                    value = value.substr(3);
                }
                data[row_count][i] = std::stod(value);
            }

            row_count++;
        }

        return data;
    }

    template<int NumDataRows>
    std::array<std::array<double, 26>, NumDataRows> readLabels(const std::string &filename) {
        std::ifstream file(filename);
        std::string line;
        std::array<std::array<double, 26>, NumDataRows> data{};

        int row_count = 0;
        while (std::getline(file, line)) {
            // Initialize the array with zeros
            std::array<double, 26> label{};
            std::fill(label.begin(), label.end(), 0.0);

            // Convert the character to its position in the alphabet and set the corresponding position in the array to 1
            int position = static_cast<int>(line[0]) - 65;
            label[position] = 1.0;

            // Store the array in the array of arrays
            data[row_count] = label;
            row_count++;
        }

        return data;
    }

    template<int NumDataColumns, int NumDataRows, int LabelSize, int K>
    void print_cross_validation_results(std::array<std::array<double, NumDataColumns>, NumDataRows> &dataset,
                                        std::array<std::array<double, LabelSize>, NumDataRows> &target,
                                        double learning_rate, double decay_rate) {
        const int fold_size = NumDataRows / K;
        double total_error = 0.0;

        // Instantiate the MultiLayerPerceptron object
        MultiLayerPerceptron<NumDataColumns, LabelSize, HIDDEN_LAYER_SIZE> mlp(learning_rate, decay_rate);

        // Create a filename based on the hyperparameters
        std::ostringstream filename_stream;
        filename_stream << "cross_validation_results_" << learning_rate << "_" << HIDDEN_LAYER_SIZE << ".txt";
        std::string filename = filename_stream.str();

        // Open the file for writing
        std::ofstream file(filename);

        for (int i = 0; i < K; ++i) {
            // Split the data into training and validation sets
            std::array<std::array<double, NumDataColumns>, fold_size> validation_data{};
            std::array<std::array<double, LabelSize>, fold_size> validation_target{};
            std::array<std::array<double, NumDataColumns>, NumDataRows - fold_size> training_data{};
            std::array<std::array<double, LabelSize>, NumDataRows - fold_size> training_target{};

            for (int j = 0; j < NumDataRows; ++j) {
                if (j >= i * fold_size && j < (i + 1) * fold_size) {
                    validation_data[j - i * fold_size] = dataset[j];
                    validation_target[j - i * fold_size] = target[j];
                } else {
                    training_data[j < i * fold_size ? j : j - fold_size] = dataset[j];
                    training_target[j < i * fold_size ? j : j - fold_size] = target[j];
                }
            }

            mlp.template train<NumDataRows - fold_size, with_momentum>(training_data, training_target);

            // Validate the model on the validation set and accumulate the error
            double fold_error = mlp.template calculate_error<fold_size>(validation_data, validation_target);
            total_error += fold_error;

            // Write the error for this fold to the file
            file << "Error for fold " << i + 1 << ": " << fold_error << "\n";

            // Reset the weights of the network
            mlp.reset_weights();
        }

        // Write the average error to the file
        file << "Average error: " << total_error / K << "\n";

        // Close the file
        file.close();
    }

    /**
 * @brief Prints the predicted and actual labels of a classification model.
 *
 * This function takes as input two arrays: `output` and `actual`.
 * The `output` array contains the predicted probabilities for each class,
 * and the `actual` array contains the one-hot encoded actual label.
 * Both arrays are assumed to be of the same size, defined by `LABEL_SIZE`.
 *
 * The function identifies the class with the highest predicted probability
 * and the actual class by finding the maximum element in the `output` and `actual` arrays, respectively.
 * The indices of these maximum elements correspond to the class labels, which are assumed to be integers from 0 to `LABEL_SIZE - 1`.
 * These indices are then converted to their corresponding ASCII characters by adding 65 (the ASCII value of 'A')
 * and cast to char to get the corresponding uppercase letter.
 *
 * The predicted and actual labels are then printed to the standard output.
 *
 * @param output An array of predicted probabilities for each class.
 * @param actual An array containing the one-hot encoded actual label.
 */
    void printPrediction(const std::array<double, LABEL_SIZE> &output, const std::array<double, LABEL_SIZE> &actual) {
        std::cout << "Predicted: "
                  << static_cast<char>(std::distance(output.begin(), std::max_element(output.begin(), output.end())) +
                                       65) << std::endl;
        std::cout << "Actual: "
                  << static_cast<char>(std::distance(actual.begin(), std::max_element(actual.begin(), actual.end())) +
                                       65) << std::endl;
    }

/**
 * @brief Generates and saves a confusion matrix from the predicted and actual results of a classification model.
 *
 * This function takes as input two 2D arrays: `predicted` and `actual`, which contain the predicted and actual labels,
 * respectively, for each data point. It also takes a string `filename`, which is the name of the file where the confusion
 * matrix will be saved.
 *
 * The function first initializes a 2D array `confusion_matrix` with zeros. This array will be used to store the confusion matrix.
 *
 * Then, it iterates over the predicted and actual labels, incrementing the corresponding element in the confusion matrix
 * for each pair of labels.
 *
 * Finally, it opens a file with the given filename and writes the confusion matrix to the file. Each row of the matrix is
 * written on a separate line, and the values in each row are separated by spaces.
 *
 * @tparam NumClasses The number of classes in the model.
 * @tparam NumData The number of data points in the dataset.
 *
 * @param predicted A 2D array containing the predicted labels for each data point.
 * @param actual A 2D array containing the actual labels for each data point.
 * @param filename The name of the file where the confusion matrix will be saved.
 */
    template<int NumClasses, int NumData>
    void generate_and_save_confusion_matrix(const std::array<std::array<double, NumClasses>, NumData> &predicted,
                                            const std::array<std::array<double, NumClasses>, NumData> &actual,
                                            const std::string &filename) {
        // Initialize confusion matrix with zeros
        std::array<std::array<int, NumClasses>, NumClasses> confusion_matrix{};
        for (auto &row: confusion_matrix) {
            row.fill(0);
        }

        // Generate confusion matrix
        for (int i = 0; i < NumData; ++i) {
            int predicted_class = std::distance(predicted[i].begin(),
                                                std::max_element(predicted[i].begin(), predicted[i].end()));
            int actual_class = std::distance(actual[i].begin(), std::max_element(actual[i].begin(), actual[i].end()));
            confusion_matrix[actual_class][predicted_class]++;
        }

        // Save confusion matrix to file
        std::ofstream file(filename);
        for (const auto &row: confusion_matrix) {
            for (const auto &value: row) {
                file << value << ' ';
            }
            file << '\n';
        }
    }

/**
 * @brief Generates predictions for a given dataset using a trained model.
 *
 * This function takes as input a trained model and a dataset, and generates predictions for each data point in the dataset.
 * The model is assumed to be a MultiLayerPerceptron object, and the dataset is assumed to be a 2D array where each row is a data point and each column is a feature.
 * The function returns a 2D array where each row corresponds to a data point and each column is the predicted probability for each class.
 *
 * @tparam NumDataColumns The number of features in the dataset.
 * @tparam NumData The number of data points in the dataset.
 * @tparam NumClasses The number of classes in the model.
 *
 * @param model A reference to the trained model.
 * @param dataset The dataset for which to generate predictions.
 *
 * @return A 2D array containing the predicted probabilities for each class for each data point in the dataset.
 */
    template<int NumDataColumns, int NumData, int NumClasses>
    std::array<std::array<double, NumClasses>, NumData>
    generate_predictions(MultiLayerPerceptron<NumDataColumns, NumClasses, HIDDEN_LAYER_SIZE> &model,
                         const std::array<std::array<double, NumDataColumns>, NumData> &dataset) {
        std::array<std::array<double, NumClasses>, NumData> predictions{};
        for (int i = 0; i < NumData; ++i) {
            predictions[i] = model.predict(dataset[i]);
        }
        return predictions;
    }

}

int main(int argc, char* argv[]) {
    auto data_letras = readData<DATA_COLUMNS, DATA_ROWS>(FILENAME_DATA);
    auto labels_letras = readLabels<DATA_ROWS>(FILENAME_LABELS);

    auto [data_letras_split, labels_letras_split] = splitData<DATA_COLUMNS, DATA_ROWS, LABEL_SIZE, SPLIT_INDEX>(
            data_letras, labels_letras);

    MultiLayerPerceptron<DATA_COLUMNS, LABEL_SIZE, HIDDEN_LAYER_SIZE> mlp_letras(LEARNING_RATE, DECAY_RATE);

    if (argc > 1) {
        // If a weights file is provided as a command-line argument, load the weights and only execute the predictions
        std::string weights_file = argv[1];
        mlp_letras.load_weights(weights_file);

        auto predicted_labels = generate_predictions<DATA_COLUMNS, DATA_ROWS, LABEL_SIZE>(mlp_letras, data_letras);
        for (int i = 0; i < DATA_ROWS; i++) {
            printPrediction(predicted_labels[i], labels_letras[i]);
        }
    } else {
        // If no weights file is provided, perform the training, generate and print the predictions, and if SAVE_ERROR is not defined, perform cross-validation
        std::cout << "Treinando o modelo:\n\n";

        if constexpr (with_early_stopping) {
            mlp_letras.train<TRAIN_DATA_SIZE, 832, with_momentum>(data_letras_split.first, labels_letras_split.first, 25);
        } else {
            mlp_letras.train<TRAIN_DATA_SIZE, with_momentum>(data_letras_split.first, labels_letras_split.first);
        }

        if constexpr (SAVE_ERROR) {
            mlp_letras.save_training_data(FILENAME_ERRORS);
        }

        auto predicted_labels = generate_predictions<DATA_COLUMNS, DATA_ROWS - TRAIN_DATA_SIZE, LABEL_SIZE>(mlp_letras,
                                                                                                            data_letras_split.second);

        std::cout << "\n\nResultados da classificação:\n\n";
        for (int i = 0; i < DATA_ROWS - TRAIN_DATA_SIZE; i++) {
            printPrediction(predicted_labels[i], labels_letras_split.second[i]);
        }

        generate_and_save_confusion_matrix<LABEL_SIZE, DATA_ROWS - TRAIN_DATA_SIZE>(predicted_labels,
                                                                                    labels_letras_split.second,
                                                                                    FILENAME_CONFUSION_MATRIX);

        if constexpr (!SAVE_ERROR) {
            print_cross_validation_results<DATA_COLUMNS, DATA_ROWS, LABEL_SIZE, K_FOLDS>(data_letras, labels_letras,
                                                                                         LEARNING_RATE,
                                                                                         DECAY_RATE);
        }

        mlp_letras.save_weights(FILENAME_WEIGHTS);
    }

    return 0;
}
