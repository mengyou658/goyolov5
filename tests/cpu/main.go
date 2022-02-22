package main

import (
	"fmt"
	"github.com/danhilltech/goyolov5"
	"image"
	"image/png"
	"os"
)

func main() {
	yolov5, err := goyolov5.NewYoloV5("person-best-v1.torchscript", goyolov5.DeviceCPU, 640, false)
	if err != nil {
		panic(err)
	}

	f, err := os.Open("1.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	input, _, err := image.Decode(f)
	if err != nil {
		panic(err)
	}
	tensor := goyolov5.NewTensorFromImage(input)

	outTensor := goyolov5.NewTensorFromImage(tensor)

	predictions, err := yolov5.Infer(tensor, 0.5, 0.4, outTensor)
	if err != nil {
		panic(err)
	}

	fmt.Println(predictions)

	f2, err := os.Create("a.png")
	if err != nil {
		panic(err)
	}
	defer f2.Close()
	if err = png.Encode(f2, outTensor); err != nil {
		panic(err)
	}

}
