# Blur detect (small)
Маленькая (очень компактная) модель для детектирования нерезких изображений. Смаз в движении, излишняя зашумленность, цифровое увеличение (бикубическое), изображения не в фокусе - все эти типы должны определятся данной моделью.

Пример использования

import model_reducer

print(get_blur_predict('image1.jpg')) 
