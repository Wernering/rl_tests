from k_armed_bandit import KBanditProblem
import matplotlib.pyplot as plt
import time


def stationary_problem():
    t1 = time.time()

    x = KBanditProblem(10, 1000, 2000, 0.1)
    y = x.copy()
    z = y.copy()

    y.change_epsilon(0.01)
    z.change_epsilon(0)

    x.experiment()
    print('x listo')
    y.experiment()
    print('y listo')
    z.experiment()
    print('z listo')

    df_x = x.results
    df_y = y.results
    df_z = z.results

    plt.plot(df_x.index, df_x['average'], label='E=0.1')
    plt.plot(df_y.index, df_y['average'])
    plt.plot(df_z.index, df_z['average'])
    plt.legend()

    print(time.time() - t1)

    plt.show()

    df_x.to_excel('resultados_1000_2000_01.xlsx')


def nonstationary_problem():
    t1 = time.time()

    x = KBanditProblem(k=10, iteration=10000, rounds=2000, epsilon=0.1, stationary=False, alpha=0.1)
    y = x.copy()
    z = y.copy()

    y.change_epsilon(0.01)
    z.change_epsilon(0)

    x.experiment()
    y.experiment()
    z.experiment()

    df_x = x.results
    df_y = y.results
    df_z = z.results

    plt.plot(df_x.index, df_x['average'], label='E=0.1')
    plt.plot(df_y.index, df_y['average'], label='E=0.01')
    plt.plot(df_z.index, df_z['average'], label='E=0')

    print(time.time() - t1)

    plt.show()


if __name__ == '__main__':
    # nonstationary_problem()
    stationary_problem()