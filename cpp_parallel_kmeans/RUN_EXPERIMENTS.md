# Запуск экспериментов и вывод метрик

## Быстрый старт

### Запуск всех экспериментов

```bash
# Базовый запуск (все реализации, включая CPU)
x64\Debug\cpp_parallel_kmeans.exe all --output results.json

# Только GPU реализации
x64\Debug\cpp_parallel_kmeans.exe all --gpu-only --output results.json

# С ограничением времени (например, 30 минут)
x64\Debug\cpp_parallel_kmeans.exe all --max-seconds 1800 --output results.json
```

## Детальное описание

### Режим `all` - Запуск всех экспериментов

Запускает все 5 экспериментов последовательно:
1. **exp1_baseline_single** - Baseline (N=100000, D=50, K=8)
2. **exp2_scaling_n** - Масштабирование по N (1000, 100000, 1000000, 5000000)
3. **exp3_scaling_d** - Масштабирование по D (2, 10, 50, 200)
4. **exp4_scaling_k** - Масштабирование по K (4, 8, 16, 32)
5. **exp5_gpu_profile** - GPU профилирование (N=1000000)

### Параметры командной строки

#### `--output <file>`
Указывает файл для сохранения результатов в формате JSON (NDJSON - Newline Delimited JSON).

**По умолчанию:** `kmeans_timing_results.json`

#### `--gpu-only`
Запускает только GPU реализации (CUDA V1-V4), пропуская CPU версии.

#### `--max-seconds <sec>`
Ограничивает максимальное время выполнения всех экспериментов в секундах.

**Пример:** `--max-seconds 1800` (30 минут)

## Запуск отдельных экспериментов

### Эксперимент 1: Baseline Single
```bash
x64\Debug\cpp_parallel_kmeans.exe experiment exp1_baseline_single --output baseline.json
```

### Эксперимент 2: Scaling N
```bash
x64\Debug\cpp_parallel_kmeans.exe experiment exp2_scaling_n --output scaling_n.json
```

### Эксперимент 3: Scaling D
```bash
x64\Debug\cpp_parallel_kmeans.exe experiment exp3_scaling_d --output scaling_d.json
```

### Эксперимент 4: Scaling K
```bash
x64\Debug\cpp_parallel_kmeans.exe experiment exp4_scaling_k --output scaling_k.json
```

### Эксперимент 5: GPU Profile
```bash
x64\Debug\cpp_parallel_kmeans.exe experiment exp5_gpu_profile --output gpu_profile.json
```

## Формат результатов

Результаты сохраняются в формате **NDJSON** (Newline Delimited JSON) - каждая строка это отдельный JSON объект.

### Структура записи результата

```json
{
  "experiment": "exp1_baseline_single",
  "implementation": "cpp_cuda_v1",
  "dataset_N": "100000",
  "dataset_D": "50",
  "dataset_K": "8",
  "T_fit_avg": "0.123456789",
  "T_fit_std": "0.001234567",
  "T_fit_min": "0.120000000",
  "T_assign_total_avg": "0.080000000",
  "T_update_total_avg": "0.040000000",
  "T_iter_total_avg": "0.120000000",
  "throughput_ops_avg": "833333.33",
  "T_transfer_avg": "0.010000000",
  "T_transfer_ratio_avg": "8.10",
  "n_iters_actual_avg": "15",
  "repeats_done": "10",
  "repeats_requested": "10"
}
```

### Метрики в результатах

- **T_fit_avg** - Среднее время выполнения fit() (секунды)
- **T_fit_std** - Стандартное отклонение времени fit()
- **T_fit_min** - Минимальное время fit()
- **T_assign_total_avg** - Среднее общее время назначения кластеров
- **T_update_total_avg** - Среднее общее время обновления центроидов
- **T_iter_total_avg** - Среднее общее время одной итерации
- **throughput_ops_avg** - Средняя пропускная способность (операций/сек)
- **T_transfer_avg** - Среднее время передачи данных (H2D + D2H)
- **T_transfer_ratio_avg** - Средний процент времени передачи от общего времени
- **n_iters_actual_avg** - Среднее количество выполненных итераций
- **repeats_done** - Количество успешно выполненных повторов
- **repeats_requested** - Запрошенное количество повторов

## Просмотр результатов

### Чтение JSON файла

#### PowerShell
```powershell
# Просмотр всех записей
Get-Content results.json | ConvertFrom-Json

# Подсчет записей
(Get-Content results.json | Measure-Object -Line).Lines

# Фильтрация по реализации
Get-Content results.json | ConvertFrom-Json | Where-Object { $_.implementation -eq "cpp_cuda_v1" }
```

#### Python скрипт для анализа
```python
import json

with open('results.json', 'r') as f:
    results = [json.loads(line) for line in f]

# Группировка по эксперименту
by_experiment = {}
for r in results:
    exp = r['experiment']
    if exp not in by_experiment:
        by_experiment[exp] = []
    by_experiment[exp].append(r)

# Вывод метрик
for exp, data in by_experiment.items():
    print(f"\n{exp}:")
    for r in data:
        print(f"  {r['implementation']}: "
              f"T_fit={r['T_fit_avg']}s, "
              f"throughput={r['throughput_ops_avg']} ops/s")
```

## Примеры использования

### 1. Полный набор экспериментов (все реализации)
```bash
x64\Debug\cpp_parallel_kmeans.exe all --output full_results.json
```

### 2. Только GPU эксперименты
```bash
x64\Debug\cpp_parallel_kmeans.exe all --gpu-only --output gpu_results.json
```

### 3. Быстрый тест с ограничением времени
```bash
x64\Debug\cpp_parallel_kmeans.exe all --max-seconds 600 --output quick_test.json
```

### 4. Комбинированный запуск
```bash
x64\Debug\cpp_parallel_kmeans.exe all --gpu-only --max-seconds 3600 --output comprehensive_gpu.json
```

## Вывод в консоль

Во время выполнения экспериментов в консоль выводится:
- Прогресс выполнения каждого эксперимента
- Информация о текущем датасете (N, D, K)
- Название запускаемой реализации
- Итоговая информация о сохранении результатов

**Пример вывода:**
```
Dataset N=100000, D=50, K=8, repeats=10
  Running cpp_cuda_v1
  Running cpp_cuda_v2
  Running cpp_cuda_v3
  Running cpp_cuda_v4
...

Результаты сохранены в: results.json
Всего записей: 45
```

## Примечания

1. **Время выполнения**: Полный набор экспериментов может занять значительное время (от 30 минут до нескольких часов в зависимости от конфигурации).

2. **Память**: Убедитесь, что у вас достаточно памяти для больших датасетов (особенно N=5000000).

3. **GPU**: Для GPU экспериментов требуется CUDA-совместимая видеокарта.

4. **Повторы**: Количество повторов (repeats) настраивается в `experiments_config.h` и может варьироваться в зависимости от размера датасета.

5. **Формат JSON**: Файл использует формат NDJSON (каждая строка - отдельный JSON), что удобно для потоковой обработки больших результатов.

