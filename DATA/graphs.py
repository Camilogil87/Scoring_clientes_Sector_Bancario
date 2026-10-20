import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate
import pandas as pd



def grouped_bar_plot(
    df, amount_key, group_key, color_key, width=0.35, title=None, amount_title=None, group_title=None,
    color_title=None, rotate_x_labels=False, aggfunc='mean'
):
    assert aggfunc in ('mean', 'sum'), "aggfunc tiene que ser sum o mean."
    
    color_scale = plt.get_cmap('Blues')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordenamos el DataFrame por la clave del grupo y del color.
    df = df.sort_values([group_key, color_key])

    # Obtenemos las etiquetas de los grupos.
    group_labels = df[group_key].unique()
    
    # Calculamos el número de grupos.
    number_of_groups = group_labels.shape[0]
    
    # Calculamos el número de barras por grupo.
    number_of_bars_per_group = df[color_key].unique().shape[0]
    
    # Transformamos el String del parámetro en una función de agregación de Numpy.
    if aggfunc == 'mean':
        aggfunc = np.mean
    elif aggfunc == 'sum':
        aggfunc = np.sum
        
    pivot_table = pd.pivot_table(
        df, values=amount_key, index=[group_key], columns=[color_key], aggfunc=aggfunc
    )
    
    # Calculamos la distancia entre los grupos.
    distance = width * (number_of_bars_per_group + 1)
    
    # Calculamos la posición de cada grupo en el eje X.
    group_x_possition = np.linspace(1, distance * number_of_groups, number_of_groups)
    
    # Iteramos los grupos y, en cada iteración, representamos todas las barras de un mismo color.
    for i, (label, values) in enumerate(pivot_table.items()):
        # Calculamos la posición relativa de la barra con respecto al grupo.
        relative_position = ((1 - number_of_bars_per_group)/2 + i) * width
        
        # Representamos las barras de un mismo color.
        ax.bar(
            group_x_possition + relative_position,
            values,
            width=width,
            color=color_scale((i+1)/number_of_bars_per_group),
            label=label
        )
    
    # Añadimos la leyenda.    
    ax.legend(bbox_to_anchor=(1.05, 1), title=color_title)

    # Etiquetamos la figura.
    fig.suptitle(title)

    ax.set_xticks(group_x_possition)
    if rotate_x_labels:
        ax.set_xticklabels(group_labels, rotation=30, ha='right')
    else:
        ax.set_xticklabels(group_labels)
        
    ax.set_ylabel(amount_title)
    ax.set_xlabel(group_title)

    # Mostramos la cuadricula.
    ax.grid(True, axis='y')

    # Ocultamos el marco.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
    return fig, ax


def ridgeline(ax, data, distributions_distance=0.5, labels=None, n_points=150, multiple_distributions=False):
    def plot_sigle_distribution(ax, distribution, hight_index, color_index=0):
        color = plt.get_cmap('tab10').colors[color_index]

        # Calculamos la altura de la distribución a representar.
        this_distributions_hight = hight_index * distributions_distance

        # Estimamos la función de densidad.
        density_function = stats.kde.gaussian_kde(distribution)

        # Calculamos la Y.
        y = density_function(x)

        # Rellenamos el área entre la Y y la altura de esta distribución.
        ax.fill_between(
            x,
            y + this_distributions_hight,
            [this_distributions_hight] * len(x),
            color=color,
            alpha=0.8
        )


    if labels is not None:
        assert len(data) == len(labels)

    # Obtenemos el valor mínimo y el máximo.
    if multiple_distributions:
        min_value = min([np.min(np.concatenate(sublist)) for sublist in data])
        max_value = max([np.max(np.concatenate(sublist)) for sublist in data])
    else:
        min_value = np.min(np.concatenate(data))
        max_value = np.max(np.concatenate(data))

    # Calculamos el rango del gráfico para calcular el tamaño del margen.
    values_range = max_value - min_value

    # Calculamos los valores de X.
    x = np.linspace(min_value - .1*values_range, max_value + .1*values_range, n_points)

    # Iteramos las distribuciones recibidas.
    for i, distribution in enumerate(data):
        # Calculamos el indice comenzando desde arriba.
        i=len(data)-i

        # Representamos cada una de las distribuciones. Si hemos recibido varias listas de distribuciones las iteramos.
        if multiple_distributions:
            for j, dist in enumerate(distribution):
                plot_sigle_distribution(ax, dist, i, j)
        else:
            plot_sigle_distribution(ax, distribution, i)

    # Si hemos recibido etiquetas, las representamos.
    if labels is not None:
        ax.set_yticks(np.arange(distributions_distance, (len(labels) + .5) *distributions_distance, distributions_distance))
        ax.set_yticklabels(reversed(labels))


def stacked_densities(
    ax, data, labels=None, n_points=150, relative=False, multiple_ax=False, x_label=None, y_label=None
):
    assert labels is None or len(labels) == len(data)
    assert not multiple_ax or len(ax) == len(data)
        
    # Creamos un el array X desde el mínimo hasta el máximo.
    x = np.linspace(np.min(np.concatenate(data)), np.max(np.concatenate(data)), n_points)
    
    # Calculamos todas las Ys.
    cached_ys = list()
    total_y = np.zeros(len(x))
    for distribution in data:
        density_function = stats.kde.gaussian_kde(distribution)
        y = density_function(x) * len(distribution)
        total_y += y
        cached_ys.append(y)
            
    # Inicializamos la y base.
    cumulative_y = np.zeros(len(x))
    
    # Representamos las densidades.
    for i, y in enumerate(cached_ys):
        actual_ax = ax[i] if multiple_ax else ax

        # Si la densidad es relativa, dividimos entre el total.
        if relative:
            y /= total_y
            
        # Obtenemos la etiqueta.
        label = labels[i] if labels else None
        
        # Si tenemos varios Axes, representamos el total en el fondo.
        if multiple_ax:
            reference_y = total_y if not relative else np.ones(len(x))
            actual_ax.plot(x, reference_y, alpha=.5, color='grey')
            actual_ax.fill_between(x, reference_y, alpha=.5, color='grey')
            
            actual_ax.set_title(label)

        # Representamos la densidad.
        actual_ax.plot(x, y + cumulative_y, label=label)
        actual_ax.fill_between(x, cumulative_y, y + cumulative_y, alpha=0.8)
        
        # Si solo hay un Axes, acumulamos el inicio de la Y.
        if not multiple_ax:
            cumulative_y += y
            
        # Etiquetamos los ejes.
        actual_ax.set_xlabel(x_label)
        
        if not multiple_ax or i == 0:
            actual_ax.set_ylabel(y_label)

        actual_ax.spines['top'].set_visible(False)
        actual_ax.spines['right'].set_visible(False)
        actual_ax.spines['bottom'].set_visible(False)
        actual_ax.spines['left'].set_visible(False)
        actual_ax.grid(True, axis='y')


def mosaic_plot(
    ax, df, first_proportion_column, second_proportion_column, color_scale=None,
    first_column_classes_order=None, second_column_classes_order=None
):
    def _get_sorted_classes(df, column, classes_order):
        if classes_order is None:
            # Si no hay orden intrinseco, se muestran de mayor a menor.
            return tuple(df[column].value_counts().index)
        else:
            column_classes = df[column].unique()
            assert set(classes_order) == set(column_classes)

            return classes_order

    if color_scale is None:
        color_scale=plt.get_cmap('tab10').colors

    # Obtenemos los diferentes valores de cada columna.
    first_column_classes = _get_sorted_classes(df, first_proportion_column, first_column_classes_order)
    second_column_classes = _get_sorted_classes(df, second_proportion_column, second_column_classes_order)

    # Añadimos una separación de 1% entre las barras.
    bar_x_distance = df.shape[0] * .01
    bar_y_distance = .01

    last_bar_x_end = 0

    x_ticks = list()
    y_ticks = list()

    # Iteramos por los diferentes valores de la primera columna.
    for first_column_class in first_column_classes:
        # Obtenemos el número de elementos pertenecientes a 'first_column_class'.
        first_column_df_subset = df[df[first_proportion_column]==first_column_class]
        first_column_df_subset_count = first_column_df_subset.shape[0]

        # Calculamos el centro de la barra.
        bar_x_position = last_bar_x_end + first_column_df_subset_count/2
        x_ticks.append(bar_x_position)

        # Calculamos el final de la barra en el eje X.
        last_bar_x_end += first_column_df_subset_count + bar_x_distance

        last_bar_y_end = 0

        # Iteramos por los diferentes valores de la segunda columna y por los colores asociados a
        #   cada uno de estos.
        for second_column_class, color in zip(second_column_classes, color_scale):
            # Obtenemos el número de elementos pertenecientes a 'first_column_class' y 'second_column_class'.
            df_subset = first_column_df_subset[
                first_column_df_subset[second_proportion_column]==second_column_class
            ]
            df_subset_count = df_subset.shape[0]

            # Calculamos la altura de la barra.
            bar_hight = df_subset_count/first_column_df_subset_count - bar_y_distance/2

            if first_column_class == first_column_classes[0]:
                # Primera iteración del bucle de fuera.
                y_ticks.append(last_bar_y_end+bar_hight/2)

            # Representamos la barra correspondiente a la porción de los elementos pertenecientes a
            #   'first_column_class' y 'second_column_class'.
            ax.bar(
                bar_x_position,
                bar_hight,
                width=first_column_df_subset_count,
                bottom=last_bar_y_end,
                color=color
            )

            # Calculamos el final de la barra en el eje Y.
            last_bar_y_end += bar_hight + bar_y_distance

    # Etiquetamos ambos ejes.
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(first_column_classes)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(second_column_classes)

    ax.set_xlim((0, last_bar_x_end))

    # Eliminamos el marco de la figura.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def nested_pie_chart(
    ax, df, first_proportion_columm, second_proportion_column, color_scale=None, label_function=None,
    pie_kwargs=None
):
    # Inicializamos los valores por defecto de los parámetros.
    if pie_kwargs is None:
        pie_kwargs = dict()

    if color_scale is None:
        color_scale=plt.get_cmap('tab10').colors

    if label_function is None:
        label_function=lambda first_column_class, second_column_class: f'{second_column_class} ({first_column_class})'

    # Obtenemos los diferentes valores de cada columna.
    first_column_classes = df[first_proportion_columm].unique()
    second_column_classes = df[second_proportion_column].unique()

    # Calculamos el alpha de los diferentes niveles de transparencia necesarios.
    color_alphas = np.linspace(.5, 1, len(second_column_classes))

    # Inicializamos las listas que debemos rellenar para representar el gráfico.
    labels = list()
    colors = list()
    proportions = list()

    # Iteramos por los diferentes valores de la primera columna y por los colores asociados a cada uno de estos.
    for first_column_class, color in zip(first_column_classes, color_scale):
        # Iteramos por los diferentes valores de la segunda columna y por el alpha asociados a cada uno de estos.
        for second_column_class, color_alpha in zip(second_column_classes, color_alphas):
            # Calculamos el número de elementos que pertenecen a ambas clases.
            proportion_count = df[
                (df[first_proportion_columm]==first_column_class) & (df[second_proportion_column]==second_column_class)
            ].shape[0]
            if proportion_count == 0:
                # Si no hay ningún elemento, no lo representamos.
                continue
            proportions.append(proportion_count)

            # Obtenemos la etiqueta para este conjunto.
            label = label_function(first_column_class, second_column_class)
            labels.append(label)

            # Obtenemos el color del conjunto mediante el color de la primera clase y la transparencia de la segunda.
            color_with_alpha = (*color, color_alpha)
            colors.append(color_with_alpha)

    ax.pie(
        proportions,
        colors=colors,
        labels=labels,
        autopct=lambda proportion: f'{(proportion/100 * df.shape[0]):.0f}',
        **pie_kwargs
    )


def parallel_set(ax, df, set_keys):
    def set_text(df, ax, set_keys):
        total = df.shape[0]
        x = 0
        # Recorremos las columnas.
        for set_key in set_keys:
            y = 1
            # Recorremos las clases de cada columna.
            for class_name in reversed(df[set_key].unique()):
                # Calculamos el tamaño del grupo.
                count = df[df[set_key] == class_name].shape[0] / total
                # Añadimos un guión al inicio del grupo.
                ax.text(x, y, '-', horizontalalignment='center', verticalalignment='center')
                y -= count / 2
                # Añadimos el nombre de la clase en el centro de su proporción.
                ax.text(x, y, class_name, horizontalalignment='center', verticalalignment='center')
                y -= count / 2
                # añadimos un quión al final del grupo.
                ax.text(x, y, '-', horizontalalignment='center', verticalalignment='center')
            x += 1

    def connect_lines(df, color_key, start_line_key, end_line_Key, ax, start_x, end_x, color_dict):
        # Calculamos el número total de filas.
        total_count = df.shape[0]

        # Creamos un diccionario que almacene la posición donde empieza cada clase en la primera columna.
        start_line_classes_start = dict()
        class_start = 1
        for class_name in reversed(df[start_line_key].unique()):
            # Calculamos el tamaño de la proporción.
            class_proportion = df[df[start_line_key] == class_name].shape[0] / total_count

            # Lo almacenamos en el diccionario.
            start_line_classes_start[class_name] = class_start

            # Actualizamos el acumulado.
            class_start -= class_proportion

        # Repetimos esta operación con los valores de la segunda columna.
        end_line_classes_start = dict()
        class_start = 1
        for class_name in reversed(df[end_line_Key].unique()):
            class_proportion = df[df[end_line_Key] == class_name].shape[0] / total_count

            end_line_classes_start[class_name] = class_start
            class_start -= class_proportion

        # Inicializamos el array X.
        x = np.arange(start_x, end_x + .1, .1)

        # Recorremos las clases del color.
        for color_class in reversed(df[color_key].unique()):
            # Filtramos el DataFrame.
            color_filtered_df = df[df[color_key] == color_class]

            # Recorremos las clases de la primera columna.
            for start_line_class in reversed(df[start_line_key].unique()):
                # Filtramos el DataFrame.
                start_line_filtered_df = color_filtered_df[color_filtered_df[start_line_key] == start_line_class]

                # Recorremos las clases de la segunda columna.
                for end_line_class in reversed(df[end_line_Key].unique()):
                    # Filtramos el DataFrame.
                    end_line_filtered_df = start_line_filtered_df[
                        start_line_filtered_df[end_line_Key] == end_line_class]

                    # Calculamos el tamaño de esta franja.
                    count = end_line_filtered_df.shape[0] / total_count

                    # Si es 0, ignoramos esta iteración.
                    if count == 0:
                        continue

                    # Interpolamos el inicio de la franja.
                    f1 = smooth_line(start_x, end_x, start_line_classes_start[start_line_class],
                                     end_line_classes_start[end_line_class])

                    # Actualizamos los diccionarios con la anchura de la franja.
                    start_line_classes_start[start_line_class] -= count
                    end_line_classes_start[end_line_class] -= count

                    # Interpolamos el final de la franja.
                    f2 = smooth_line(start_x, end_x, start_line_classes_start[start_line_class],
                                     end_line_classes_start[end_line_class])

                    # Representamos la franja.
                    ax.fill_between(x, f1(x), f2(x), alpha=0.6, color=color_dict[color_class])

    def smooth_line(x1, x2, y1, y2):
        # Añadimos otro punto a la derecha e izquierda de los originales.
        a = .1
        x = np.array([x1 - a, x1, x2, x2 + a])
        y = np.array([y1, y1, y2, y2])
        # Interpolamos una línea que une los 4 puntos con una función cuadrática.
        f = interpolate.interp1d(x, y, kind='quadratic')
        return f

    assert len(set_keys) > 1

    # Creamos un diccionario con los colores para cada clave.
    color_scale = plt.get_cmap('tab10').colors

    color_dict = dict()

    color_key = set_keys[0]
    for i, color_class in enumerate(df[color_key].unique()):
        color_dict[color_class] = color_scale[i]

    # Añadimos las etiqueras en cada columna.
    set_text(df, ax, set_keys)

    # Creamos las líneas que unen dos columnas.
    for i in range(len(set_keys) - 1):
        start_key = set_keys[i]
        end_key = set_keys[i + 1]

        connect_lines(df, color_key, start_key, end_key, ax, i, i + 1, color_dict)

    # Etiquetamos el eje X.
    for i, set_key in enumerate(set_keys):
        ax.text(i, -.075, set_key, ha='center')

    # Ocultamos el marco.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_correlogram(fig, ax, df, correlated_variables):
    # Filtramos el DataFrame con las variables introducidas.
    filtered_df = df[correlated_variables]

    # Obtenemos la matriz de correlación.
    correlation_coefficients = filtered_df.corr()

    # Representamos la matriz.
    im = ax.imshow(correlation_coefficients, vmin=-1, vmax=1, cmap='PiYG')

    # Representamos la leyenda de color.
    fig.colorbar(im)

    # Etiquetamos los ejes.
    ticks = np.arange(len(correlation_coefficients))

    ax.set_yticks(ticks)
    ax.set_yticklabels(correlated_variables)

    ax.set_xticks(ticks)
    ax.set_xticklabels(correlated_variables, rotation=30, ha='right')


def plot_bubble_correlogram(fig, ax, df, correlated_variables):
    filtered_df = df[correlated_variables]
    correlation_coefficients = filtered_df.corr()
    
    x = list()
    y = list()
    color = list()
    size = list()
    
    # Istanciamos los arrays 'x', 'y', 'color' y 'size' con las coordenadas y las valores.
    for i, row in enumerate(correlation_coefficients.values):
        for j, cell in enumerate(reversed(row)):
            x.append(i)
            y.append(j)
            color.append(cell)
            size.append(abs(cell)*1000)
            
    # Representamos el correlograma.
    sc = ax.scatter(x=x, y=y, s=size, c=color, cmap='PiYG', vmin=-1, vmax=1)
    
    # Añadimos la leyenda de color.
    fig.colorbar(sc)

    # Etiquetamos los ejes.
    ticks = np.arange(len(correlation_coefficients))
    ax.set_yticks(ticks)
    ax.set_yticklabels(reversed(correlated_variables))
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(correlated_variables, rotation=30, ha='right')
    
    ## Ocultamos el marco.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_connected_scatter(ax, x, y, same_scale=False, gradient_count=100, cmap='Blues'):
    color_scale = plt.get_cmap(cmap)

    # Calculamos el número de puntos de cada zona coloreada.
    jump_size = x.shape[0]/gradient_count

    # Obtenemos un array entre 0.1 y 1 con tantos números como colores a representar.
    color_gradients = np.linspace(0.1, 1, gradient_count)

    for i, color_gradient in enumerate(color_gradients):
        # Obtenemos los puntos del inicio y final del slice.
        slice_start = int(i*jump_size)
        slice_end = int((i+1)*jump_size) +1

        # Representamos esos puntos con el color correspondiente.
        ax.plot(
            x.values[slice_start:slice_end],
            y.values[slice_start:slice_end],
            color=color_scale(color_gradient)
        )


    # Etiquetamos la fecha de seis puntos equidistantes.
    for i in np.linspace(0, x.shape[0]-1, 6):
        i = int(i)
        ax.text(
            x.iloc[i],
            y.iloc[i],
            x.index.to_series().iloc[i].date()
        )

    # Si el parámetro 'same_scale', hacemos que ambos ejes tengan la misma escala.
    if same_scale:
        # Obtenemos el rango de valores de X e Y.
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)

        # Reservamos el mayor de los dos.
        bigger_range = max(x_range, y_range)

        # Obtenemos un margen del 10%
        margin = bigger_range * .1

        # Calculamos el punto medio de 'range_x' y 'range_y'.
        x_middle = np.min(x) + x_range/2
        y_middle = np.min(y) + y_range/2

        # Añadimos una escala del tamaño del mayor rango y centrado en el punto médio de cada eje.
        ax.set_xlim([
            x_middle - bigger_range/2 - margin,
            x_middle + bigger_range/2 + margin
        ])
        ax.set_ylim([
            y_middle - bigger_range/2 - margin,
            y_middle + bigger_range/2 + margin
        ])

    ## Mostramos la cuadricula.
    ax.grid(True)

    ## Ocultamos el marco.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
