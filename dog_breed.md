### Import data

```bash
import kagglehub
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
print("Path to dataset files:", path)
```

Download data from kaggle.

### Data preprocessing

```bash
image_size = 299  # Required input size for InceptionResNetV2
batch_size = 32

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # 20% validation split for train and validation
)
```

batch_size = 32 means that during training, the model processes 32 images in each batch (mini-batch) before updating the gradients.

`ImageDataGenerator` is a data augmentation and preprocessing tool in Keras. It automatically normalizes images, resizes them, and applies transformations to improve the model's generalization ability.

`preprocess_input` is a special preprocessing function for InceptionResNetV2, This helps to speed up convergence and improve model training performance.

```bash
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Shuffle data for training
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
```

This initializes a data generator for the training set, which automatically loads and preprocesses images from the directory.

And similar for validation set.

```bash
# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
```

Extracts the last feature maps from the base model.

Applies Global Average Pooling (GAP), which averages each feature map across all spatial locations. It can reduces overfitting by avoiding too many parameters.

Adds a fully connected layer with 1024 neurons and ReLU activation. It captures complex patterns learned from the base model.

Final classification layer with softmax activation. Softmax can converts outputs into class probabilities and ensures that the sum of all output probabilities is 1.

```bash
model = Model(inputs=base_model.input, outputs=predictions)
```

Create our full model by using base model's input and our custom layers.

```bash
for layer in base_model.layers:
    layer.trainable = False
```

Freezes all layers in the pretrained InceptionResNetV2 model so their weights won’t be updated during training.

Retains learned features from ImageNet and prevents overfitting when training on small datasets.

```bash
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```

Uses Stochastic Gradient Descent (SGD) with a learning rate of 0.001 for training.

More stable updates compared to Adam and generalizes better for fine-tuning models.

Uses categorical cross-entropy because we are dealing with multi-class classification.

### Train the model

```bash
train_steps = train_generator.samples // train_generator.batch_size
val_steps = val_generator.samples // val_generator.batch_size
```

Computes how many batches (steps) are needed to go through the entire training dataset once.

Example: If we have 10,000 training images and a batch size of 32, we need 312 steps to complete one epoch.

Simialr to the validation dataset.

```bash
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,  # Explicitly set the correct steps
    validation_data=val_dataset,
    validation_steps=val_steps,
    epochs=10,
    verbose=1
)
```

Train the model and store the result in history

history.history['loss'] -> train loss
history.history['accuracy'] -> train accuracy
history.history['val_loss'] ->  test loss
history.history['val_accuracy'] -> test accuracy

### Fine-Tuning the model

```bash
for layer in base_model.layers[:200]:
    layer.trainable = False
for layer in base_model.layers[200:]:
    layer.trainable = True
```

This unfreezes part of the InceptionResNetV2 model so that it can be fine-tuned on our dataset.

Initially, we froze all layers to only train the new classification layers. Now, we unfreeze the deeper layers (from layer 200 onward) so they can be fine-tuned.

First 200 layers → Keep frozen (these layers learn common features such as edges, textures)
Later layers → Unfreeze and train (these layers learn more advanced features such as class-specific patterns)

```bash
model.compile(optimizer=SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

Since we changed the trainable layers, we need to recompile the model before training again.



```bash
fine_tune_epochs = 10
total_epochs = 10 + fine_tune_epochs

history_fine_tune = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,  # Same steps as initial training
    validation_data=val_dataset,
    validation_steps=val_steps,
    epochs=total_epochs,
    initial_epoch=10
)
```

This continues training the model for 10 more epochs with fine-tuning enabled.

First 10 epochs → train new classification layers (improve model generalization ability)
Last 10 epochs → fine-tune deep features (improve accuracy)
