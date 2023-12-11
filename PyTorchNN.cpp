#include <SFML/Graphics.hpp>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>

struct DigitProgress {
    sf::Text label;
    sf::RectangleShape background;
    sf::RectangleShape progress;
    sf::Text progressLabel;
};

struct Button {
    sf::RectangleShape shape;
    sf::Text text;
};

struct Label {
    sf::Text text;
};

std::vector<std::string> read_csv_lines(const std::string& filename, int num_lines) {
    std::vector<std::string> lines;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        exit(1);
    }
    // Read specified number of lines from file
    std::string line;
    // Ignore magic numbers
    getline(infile, line);
    for (int i = 0; i < num_lines && getline(infile, line); i++) {
        lines.push_back(line);
    }
    infile.close();
    return lines;
}

void make_images_from_mnist_dataset(std::vector<std::string> lines) {
    // Размер изображения MNIST
    int rows = 28, cols = 28;

    // Увеличенный размер окна imshow
    int displaySize = 400;

    // Создаем матрицу для хранения изображения
    cv::Mat image(rows, cols, CV_8UC1);
    // Обрабатываем каждую строку
    for (const std::string& line : lines) {
        // Первое число в строке - метка класса
        // Игнорируем его
        std::stringstream ss(line);
        std::string label_str;
        getline(ss, label_str, ',');
        int label = stoi(label_str);
        // Считываем оставшиеся числа в строке и заполняем матрицу изображения
        std::string pixel_value;
        int row = 0, col = 0;
        while (getline(ss, pixel_value, ',')) {
            image.at<uchar>(row, col) = stoi(pixel_value);
            col++;
            if (col >= cols) {
                col = 0;
                row++;
            }
        }
        // Увеличение размера окна imshow
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(displaySize, displaySize));

        // Отображение изображения
        imshow("MNIST Image", resizedImage);
        cv::waitKey(0);
    }
}

void mnist_checking_form() {
    // Имя csv-файла с данными MNIST
    std::string filename = "mnist_train.csv";
    // Количество строк для чтения
    int num_lines = 100;
    // Считываем указанное количество строк из файла
    std::vector<std::string> lines = read_csv_lines(filename, num_lines);
    make_images_from_mnist_dataset(lines);
}

void training_and_testing() {
    system("cd F:\\Projects\\c++\\modularNN\\PyTorchNN && python.exe main.py");
}

void predict_digit(std::vector<float> pixels, torch::jit::script::Module module, std::vector<DigitProgress>& digitProgresses) {
    // Convert the input pixels to a Torch tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor inputTensor = torch::from_blob(pixels.data(), { 1, 1, 28, 28 }, options);

    // Set the module to evaluation mode
    module.eval();

    // Run the forward pass
    torch::Tensor outputTensor = module.forward({ inputTensor }).toTensor();

    // Apply the softmax activation
    torch::Tensor probabilities = torch::softmax(outputTensor, 1);

    // Get the predicted digit and the output values for all digits
    std::map<int, float> digitOutputs;
    auto probabilitiesData = probabilities.accessor<float, 2>();
    for (int i = 0; i < probabilities.size(1); ++i) {
        float outputValue = probabilitiesData[0][i];
        digitOutputs[i] = outputValue;
    }

    int counter = 0;
    // Print the output values for all digits
    std::cout << "Output Probabilities for Each Digit:" << std::endl;
    for (const auto& entry : digitOutputs) {
        std::cout << "Digit " << entry.first << ": " << entry.second << std::endl;
        digitProgresses[counter].progressLabel.setString(std::to_string(static_cast<int>(entry.second*100)) + "%");

        float progressWidth = entry.second*100 * 2.f;
        digitProgresses[counter].progress.setSize(sf::Vector2f(progressWidth, 20.f));
        counter++;
    }

    // Get the predicted digit
    int predictedDigit = probabilities.argmax(1).item<int>();

    // Print the predicted digit
    std::cout << "Predicted Digit: " << predictedDigit << std::endl;
}

void button_1_clicked(int width, int height) {

    std::cout << "Загрузка модели..." << std::endl;

    std::string modelPath = "F:/Projects/c++/modularNN/PyTorchNN/multimodular_network.pt";
    torch::jit::script::Module module;

    try {
        // Load the serialized model
        module = torch::jit::load(modelPath);
        std::cout << "Модель модульной нейронной сети загружена успешно!" << std::endl;

        // Use the loaded model for inference or further processing
        // ...
    }
    catch (const c10::Error& e) {
        std::cerr << "Ошибка загрузки модели: " << e.msg() << std::endl;
    }

    sf::RenderWindow window(sf::VideoMode(width, height), "Testing model", sf::Style::Titlebar | sf::Style::Close);

    sf::Font font;
    if (!font.loadFromFile("fonts/Montserrat-bold.ttf")) {
        std::cout << "Error loading font!" << std::endl;
    }

    //window.setFramerateLimit(120); // Set the frame rate limit

    // Calculate the size of the container based on the desired internal area
    int containerSize = std::min(width * 0.5f, height * 0.8f); // 50% of the window width, up to 80% of the window height

    sf::RectangleShape container(sf::Vector2f(containerSize, containerSize));
    container.setFillColor(sf::Color::Black);
    container.setOutlineThickness(2.0f);
    container.setOutlineColor(sf::Color::Magenta);

    // Position the container on the left side of the window
    float posX = 0.1f * width; // 10% margin from the left edge of the window
    float posY = (height - containerSize) / 2.0f;
    container.setPosition(posX, posY);

    std::vector<Button> buttons(2);
    buttons[0].text.setString("Predict digit");
    buttons[0].text.setFont(font);
    buttons[0].text.setCharacterSize(18);
    buttons[0].text.setFillColor(sf::Color::Black);
    buttons[0].text.setPosition(posX + containerSize + 80, posY + 10);

    buttons[0].shape.setSize(sf::Vector2f(150, 40));
    buttons[0].shape.setFillColor(sf::Color(200, 200, 200));
    buttons[0].shape.setPosition(posX + containerSize + 70, posY);

    buttons[1].text.setString("Clear");
    buttons[1].text.setFont(font);
    buttons[1].text.setCharacterSize(18);
    buttons[1].text.setFillColor(sf::Color::Black);
    buttons[1].text.setPosition(posX + containerSize / 2 - 30, posY + containerSize + 50);

    buttons[1].shape.setSize(sf::Vector2f(150, 40));
    buttons[1].shape.setFillColor(sf::Color(200, 200, 200));
    buttons[1].shape.setPosition(posX + containerSize / 2 - 40, posY + containerSize + 40);

    std::vector<sf::RectangleShape> pixels;

    std::vector<std::vector<float>> input_pixels(28, std::vector<float>(28));
    for (int i = 0; i < input_pixels.size(); i++) {
        for (int k = 0; k < input_pixels[i].size(); k++) {
            input_pixels[i][k] = 0.0f;
        }
    }

    std::vector<DigitProgress> digitProgresses(10);

    // Initialize progress bar objects for each digit
    for (int i = 0; i < digitProgresses.size(); ++i) {
        DigitProgress& digitProgress = digitProgresses[i];

        digitProgress.label.setString("This digit is: " + std::to_string(i));
        digitProgress.label.setFont(font);
        digitProgress.label.setCharacterSize(14);
        digitProgress.label.setFillColor(sf::Color::Black);
        digitProgress.label.setPosition(posX + containerSize + 70, posY + 50 + 60 * i);

        digitProgress.background.setSize(sf::Vector2f(200.f, 20.f));
        digitProgress.background.setFillColor(sf::Color::White);
        digitProgress.background.setOutlineThickness(1.f);
        digitProgress.background.setOutlineColor(sf::Color::Black);
        digitProgress.background.setPosition(posX + containerSize + 70, posY + 70 + 60 * i);

        digitProgress.progress.setSize(sf::Vector2f(0.f, 20.f));
        digitProgress.progress.setFillColor(sf::Color::Green);
        digitProgress.progress.setPosition(posX + containerSize + 70, posY + 70 + 60 * i);

        digitProgress.progressLabel.setString("0%");
        digitProgress.progressLabel.setFont(font);
        digitProgress.progressLabel.setCharacterSize(14);
        digitProgress.progressLabel.setFillColor(sf::Color::Black);
        digitProgress.progressLabel.setPosition(posX + containerSize + 70 + 10, posY + 70 + 60 * i);
    }

    float totalProgress = 0.f; // Total progress from 0 to 100

    bool isDrawing = false; // Flag to track drawing state

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2f mousePosition = sf::Vector2f(event.mouseButton.x, event.mouseButton.y);
                    if (container.getGlobalBounds().contains(mousePosition)) {
                        // Start drawing
                        isDrawing = true;
                    }
                    else if (buttons[1].shape.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
                        // Clear the drawings
                        for (int i = 0; i < input_pixels.size(); i++) {
                            for (int k = 0; k < input_pixels[i].size(); k++) {
                                input_pixels[i][k] = 0.0f;
                            }
                        }
                        pixels.clear();
                    }
                    else if (buttons[0].shape.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
                        std::vector<float> inputImage;
                        for (const auto& row : input_pixels) {
                            for (const auto& pixel : row) {
                                // Normalize the pixel value to the range [-1, 1] as done in the Python code
                                float normalizedPixel = (pixel / 255.0f - 0.5f) / 0.5f;
                                inputImage.push_back(normalizedPixel);
                            }
                        }
                        predict_digit(inputImage, module, digitProgresses);
                    }
                }
            }
            else if (event.type == sf::Event::MouseButtonReleased) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // Stop drawing
                    isDrawing = false;
                }
            }
            else if (event.type == sf::Event::MouseMoved) {
                if (isDrawing) {
                    sf::Vector2f mousePosition = sf::Vector2f(event.mouseMove.x, event.mouseMove.y);
                    if (container.getGlobalBounds().contains(mousePosition)) {
                        // Calculate the position of the mouse relative to the container
                        sf::Vector2f relativePosition = mousePosition - container.getPosition();

                        // Calculate the size of each pixel based on the container size
                        float pixelSize = containerSize / 28.0f; // Assuming 28x28 pixels

                        // Calculate the index of the pixel grid
                        int pixelX = relativePosition.x / pixelSize;
                        int pixelY = relativePosition.y / pixelSize;
                        //std::cout << "X: " << pixelX << " Y: " << pixelY << std::endl;
                        // Create a white pixel at the calculated position
                        sf::RectangleShape pixel(sf::Vector2f(pixelSize, pixelSize));
                        pixel.setFillColor(sf::Color::White);
                        pixel.setPosition(container.getPosition() + sf::Vector2f(pixelX * pixelSize, pixelY * pixelSize));

                        // Add the pixel to the vector
                        pixels.push_back(pixel);

                        input_pixels[pixelY][pixelX] = 255.0f;
                    }
                }
            }
        }

        //// Update the progress bar
        //totalProgress += 0.1f; // Increment the progress (you can change this value as per your requirements)

        //for (auto& digitProgress : digitProgresses) {
        //    digitProgress.progressLabel.setString(std::to_string(static_cast<int>(totalProgress)) + "%");

        //    // Ensure progress stays within the range of 0 to 100
        //    if (totalProgress > 100.f) {
        //        totalProgress = 100.f;
        //    }

        //    // Update the width of the progress shape based on the total progress
        //    float progressWidth = totalProgress * 2.f;
        //    digitProgress.progress.setSize(sf::Vector2f(progressWidth, 20.f));
        //}

        window.clear(sf::Color::White);
        window.draw(container);

        // Draw the pixels
        for (const auto& pixel : pixels) {
            window.draw(pixel);
        }

        // Draw the progress bars for each digit
        for (const auto& digitProgress : digitProgresses) {
            window.draw(digitProgress.background);
            window.draw(digitProgress.label);
            window.draw(digitProgress.progress);
            window.draw(digitProgress.progressLabel);
        }
        for (const auto& button : buttons) {
            window.draw(button.shape);
            window.draw(button.text);
        }

        window.display();
    }
}




int main()
{
    setlocale(LC_ALL, "Russian");

    sf::RenderWindow window(sf::VideoMode(400, 250), "Main Form");

    //window.setFramerateLimit(60); // Set the frame rate limit

    sf::Font font;
    if (!font.loadFromFile("fonts/Montserrat-bold.ttf")) {
        std::cout << "Ошибка загрузки шрифта." << std::endl;
        return -1;
    }

    sf::Text buttonText1("Open New Form", font, 14);
    buttonText1.setFillColor(sf::Color::Black);
    buttonText1.setPosition(30, 80);

    sf::Text buttonText2("Execute Training or Testing script", font, 14);
    buttonText2.setFillColor(sf::Color::Black);
    buttonText2.setPosition(30, 130);
    
    sf::Text buttonText3("Check MNIST dataset images", font, 14);
    buttonText3.setFillColor(sf::Color::Black);
    buttonText3.setPosition(30, 180);

    sf::RectangleShape button1(sf::Vector2f(140, 40));
    button1.setFillColor(sf::Color(200, 200, 200));
    button1.setPosition(20, 70);

    sf::RectangleShape button2(sf::Vector2f(300, 40));
    button2.setFillColor(sf::Color(200, 200, 200));
    button2.setPosition(20, 120);
    
    sf::RectangleShape button3(sf::Vector2f(240, 40));
    button3.setFillColor(sf::Color(200, 200, 200));
    button3.setPosition(20, 170);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
                    if (button1.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
                        button_1_clicked(1300, 1000);
                        // Add your code to open a new form here
                    }
                    else if (button2.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
                        training_and_testing();
                    }
                    else if (button3.getGlobalBounds().contains(mousePosition.x, mousePosition.y)) {
                        mnist_checking_form();
                    }
                }
            }
        }

        window.clear(sf::Color::White);
        window.draw(button1);
        window.draw(buttonText1);
        window.draw(button2);
        window.draw(buttonText2);
        window.draw(button3);
        window.draw(buttonText3);
        window.display();
    }

    return 0;
}
