# Blur detect (small)
Маленькая (очень компактная) модель для детектирования нерезких изображений. Смаз в движении, излишняя зашумленность, цифровое увеличение (бикубическое), изображения не в фокусе - все эти типы должны определятся данной моделью.

Пример использования
```py
import model_reducer
print(model_reducer.get_blur_predict('image1.jpg')) 
```

О разработке и экспериментах можно почитать тут: https://imageman72.livejournal.com/49930.html

# Blur detect (small)
Small (very compact) model for detecting blurred images. Motion blur, excessive noise, digital zoom (bicubic), out-of-focus images - all these types should be detected by this model.

Example of use
```py
import model_reducer
print(model_reducer.get_blur_predict('image1.jpg'))) 
```

You can read about development and experiments here: https://imageman72.livejournal.com/49930.html.

