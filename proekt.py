import numpy as np

# Генерация данных с выбросами
np.random.seed(42)  # для воспроизводимости

# 1. Создаем обычные продажи (27 дней от 200 до 1000)
regular_sales = np.random.randint(200, 1001, size=27)
outliers = np.array([5000, 7500, 12000])  # 3 дня с выбросами

# Объединяем и перемешиваем
daily_sales = np.concatenate([regular_sales, outliers])
np.random.shuffle(daily_sales)

print("=" * 70)
print("ИСХОДНЫЕ ДАННЫЕ О ПРОДАЖАХ")
print("=" * 70)
print("Массив продаж (30 дней):")
print(daily_sales)
print(f"\nРазмер массива: {daily_sales.shape}")
print(f"Количество элементов: {len(daily_sales)}")

# 2. Рассчитываем статистики
mean_sales = np.mean(daily_sales)
std_sales = np.std(daily_sales)
median_sales = np.median(daily_sales)
q1 = np.percentile(daily_sales, 25)
q3 = np.percentile(daily_sales, 75)
iqr = q3 - q1

print("\n" + "=" * 70)
print("СТАТИСТИКА ДО ОБРАБОТКИ")
print("=" * 70)
print(f"Среднее (mean): {mean_sales:.2f}")
print(f"Медиана (median): {median_sales:.2f}")
print(f"Стандартное отклонение (std): {std_sales:.2f}")
print(f"Минимум: {np.min(daily_sales)}")
print(f"Максимум: {np.max(daily_sales)}")
print(f"Квартиль 25% (Q1): {q1:.2f}")
print(f"Квартиль 75% (Q3): {q3:.2f}")
print(f"Межквартильный размах (IQR): {iqr:.2f}")

# 3. Поиск выбросов по правилу "трех сигм"
sigma_threshold = 3 * std_sales
outliers_3sigma = np.where(np.abs(daily_sales - mean_sales) > sigma_threshold)[0]

print("\n" + "=" * 70)
print("МЕТОД ТРЕХ СИГМ")
print("=" * 70)
print(f"Порог (3 * std): {sigma_threshold:.2f}")
print(f"Найдено выбросов: {len(outliers_3sigma)}")

if len(outliers_3sigma) > 0:
    print("Индексы выбросов:", outliers_3sigma)
    print("Значения выбросов:", daily_sales[outliers_3sigma])

# 4. Поиск выбросов по правилу IQR
# ⚠️ ВОТ ЭТИ СТРОКИ БЫЛИ ПРОПУЩЕНЫ! ⚠️
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = np.where((daily_sales < lower_bound) | (daily_sales > upper_bound))[0]

print("\n" + "=" * 70)
print("МЕТОД IQR (МЕЖКВАРТИЛЬНОГО РАЗМАХА)")
print("=" * 70)
print(f"Нижняя граница: {lower_bound:.2f}")
print(f"Верхняя граница: {upper_bound:.2f}")
print(f"Найдено выбросов: {len(outliers_iqr)}")

if len(outliers_iqr) > 0:
    print("Индексы выбросов:", outliers_iqr)
    print("Значения выбросов:", daily_sales[outliers_iqr])

# 5. Сравнение методов
print("\n" + "=" * 70)
print("СРАВНЕНИЕ МЕТОДОВ")
print("=" * 70)
print(f"Метод 3-х сигм нашел индексы: {outliers_3sigma}")
print(f"Метод IQR нашел индексы:      {outliers_iqr}")

if set(outliers_3sigma) == set(outliers_iqr):
    print("✅ Методы полностью совпали!")
else:
    print("❌ Методы НЕ совпали")

# 6. Замена выбросов на медиану
daily_sales_clean = daily_sales.copy()
median_value = np.median(daily_sales)

print(f"\nМедиана для замены: {median_value:.2f}")

# Заменяем выбросы, найденные методом IQR
for idx in outliers_iqr:
    daily_sales_clean[idx] = median_value

print("\n" + "=" * 70)
print("ДАННЫЕ ПОСЛЕ ЗАМЕНЫ ВЫБРОСОВ")
print("=" * 70)
print("Очищенный массив:")
print(daily_sales_clean)

# 7. Итоговый отчет
print("\n" + "=" * 70)
print("ИТОГОВЫЙ ОТЧЕТ: СРАВНЕНИЕ ДО И ПОСЛЕ")
print("=" * 70)
print(f"{'Показатель':<20} | {'До обработки':>15} | {'После обработки':>15}")
print("-" * 70)

print(f"{'Среднее (mean)':<20} | {np.mean(daily_sales):>15.2f} | {np.mean(daily_sales_clean):>15.2f}")
print(f"{'Медиана (median)':<20} | {np.median(daily_sales):>15.2f} | {np.median(daily_sales_clean):>15.2f}")
print(f"{'Стд отклонение':<20} | {np.std(daily_sales):>15.2f} | {np.std(daily_sales_clean):>15.2f}")
print(f"{'Минимум':<20} | {np.min(daily_sales):>15.2f} | {np.min(daily_sales_clean):>15.2f}")
print(f"{'Максимум':<20} | {np.max(daily_sales):>15.2f} | {np.max(daily_sales_clean):>15.2f}")