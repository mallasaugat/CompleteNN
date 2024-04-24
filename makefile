CXX = g++
CXXFLAGS = -std=c++11
LIBS = -L/opt/homebrew/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_videoio -lopencv_videostab -lopencv_objdetect -lopencv_dnn -lopencv_ml
INCLUDES = -I/opt/homebrew/include/opencv4


neuralNet: neuralNet.cpp datasetMnsit.cpp model.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

neuralNetEval: neuralNetEval.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

neuralTest: neuralTest.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

neuralTestCNN: neuralTestCNN.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

neuralNetEval: neuralNetEval.cpp datasetMnsit.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)


.PHONY: clean

clean:
	rm -f neuralNet
	rm -f neuralTest
