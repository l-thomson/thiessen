# API reference

## Outcome families

::: thiessen.gaussian

::: thiessen.probit

## Experimental outcome families

Compiled only into an extension built with the core's `experimental`
feature; a configuration naming one otherwise raises
`RequiresFeatureError`.

::: thiessen.tobit

::: thiessen.aft

::: thiessen.interval_censored

::: thiessen.ordinal

::: thiessen.student_t

::: thiessen.laplace

## Parameter groups

::: thiessen.TermParams

::: thiessen.GeometryParams

::: thiessen.StructureParams

::: thiessen.CellParams

## Experimental component options

Compiled only into an extension built with the core's `experimental`
feature; a configuration naming one otherwise raises
`RequiresFeatureError`. The experimental distance metrics are entries of
`GeometryParams(metric=)` and the linear cell basis is
`CellParams(basis="linear")`.

::: thiessen.soft_membership

::: thiessen.weighted_inclusion

::: thiessen.dart_inclusion

## Model

::: thiessen.Model

## FittedModel

::: thiessen.FittedModel

## AddiVortesRegressor

::: thiessen.estimators.AddiVortesRegressor

## AddiVortesClassifier

::: thiessen.estimators.AddiVortesClassifier

## XAddiVortesRegressor

::: thiessen.estimators.XAddiVortesRegressor

## Sampler

::: thiessen.sampler.Sampler

## Exceptions

::: thiessen.ThiessenError
