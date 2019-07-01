/****************************************************************************
** Meta object code from reading C++ file 'thresholdselector.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../thresholdselector.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'thresholdselector.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ThresholdSelector_t {
    QByteArrayData data[14];
    char stringdata0[155];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ThresholdSelector_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ThresholdSelector_t qt_meta_stringdata_ThresholdSelector = {
    {
QT_MOC_LITERAL(0, 0, 17), // "ThresholdSelector"
QT_MOC_LITERAL(1, 18, 12), // "lowThreshold"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 13), // "highThreshold"
QT_MOC_LITERAL(4, 46, 11), // "thresholdL0"
QT_MOC_LITERAL(5, 58, 9), // "threshold"
QT_MOC_LITERAL(6, 68, 11), // "thresholdL1"
QT_MOC_LITERAL(7, 80, 11), // "thresholdL2"
QT_MOC_LITERAL(8, 92, 11), // "thresholdH0"
QT_MOC_LITERAL(9, 104, 11), // "thresholdH1"
QT_MOC_LITERAL(10, 116, 11), // "thresholdH2"
QT_MOC_LITERAL(11, 128, 10), // "thresholdL"
QT_MOC_LITERAL(12, 139, 4), // "band"
QT_MOC_LITERAL(13, 144, 10) // "thresholdH"

    },
    "ThresholdSelector\0lowThreshold\0\0"
    "highThreshold\0thresholdL0\0threshold\0"
    "thresholdL1\0thresholdL2\0thresholdH0\0"
    "thresholdH1\0thresholdH2\0thresholdL\0"
    "band\0thresholdH"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ThresholdSelector[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   64,    2, 0x06 /* Public */,
       3,    2,   69,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,   74,    2, 0x08 /* Private */,
       6,    1,   77,    2, 0x08 /* Private */,
       7,    1,   80,    2, 0x08 /* Private */,
       8,    1,   83,    2, 0x08 /* Private */,
       9,    1,   86,    2, 0x08 /* Private */,
      10,    1,   89,    2, 0x08 /* Private */,
      11,    2,   92,    2, 0x08 /* Private */,
      13,    2,   97,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Float,    2,    2,
    QMetaType::Void, QMetaType::Int, QMetaType::Float,    2,    2,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   12,    5,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   12,    5,

       0        // eod
};

void ThresholdSelector::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ThresholdSelector *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->lowThreshold((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: _t->highThreshold((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 2: _t->thresholdL0((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->thresholdL1((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->thresholdL2((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->thresholdH0((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->thresholdH1((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->thresholdH2((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->thresholdL((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: _t->thresholdH((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ThresholdSelector::*)(int , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ThresholdSelector::lowThreshold)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ThresholdSelector::*)(int , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ThresholdSelector::highThreshold)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ThresholdSelector::staticMetaObject = { {
    &QMainWindow::staticMetaObject,
    qt_meta_stringdata_ThresholdSelector.data,
    qt_meta_data_ThresholdSelector,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ThresholdSelector::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ThresholdSelector::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ThresholdSelector.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "Controller"))
        return static_cast< Controller*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int ThresholdSelector::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void ThresholdSelector::lowThreshold(int _t1, float _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ThresholdSelector::highThreshold(int _t1, float _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
