import boto3
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def detect_labels(photo, bucket):
    client = boto3.client('rekognition', region_name='ap-south-1')

    # Detect labels with a maximum of 10 labels
    response = client.detect_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}}, MaxLabels=10)
    return response['Labels']

def show_bounding_boxes(photo, bucket, labels):
    s3_client = boto3.client('s3')
    
    # Download the image from S3
    s3_client.download_file(bucket, photo, 'downloaded_image.jpg')
    image = Image.open('downloaded_image.jpg')

    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes for each label with instances
    for label in labels:
        for instance in label.get('Instances', []):  # Use .get() to avoid KeyError if 'Instances' key is missing
            box = instance['BoundingBox']
            left = img_width * box['Left']
            top = img_height * box['Top']
            width = img_width * box['Width']
            height = img_height * box['Height']
            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)
            )
            draw.line(points, fill='#00d400', width=2)

    # Display the image with bounding boxes
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    bucket = 'image-labels-bucket'  # Your S3 bucket name
    photo = 'Dog Image.jpg'         # Your image file name
    
    # Detect labels
    labels = detect_labels(photo, bucket)
    
    # Filter labels with high confidence
    high_confidence_labels = [label for label in labels if label['Confidence'] > 80]  # Adjust confidence threshold as needed
    
    # Print detected labels with high confidence
    print("Detected Labels with High Confidence:")
    for label in high_confidence_labels:
        print(f"Label: {label['Name']}, Confidence: {label['Confidence']}")
    
    # Show bounding boxes for high confidence labels
    show_bounding_boxes(photo, bucket, high_confidence_labels)

if __name__ == "__main__":
    main()
